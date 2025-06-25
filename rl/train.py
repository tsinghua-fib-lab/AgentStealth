# -*- coding: utf-8 -*-
import torch
import random
import logging
import os
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from src.utils.string_utils import compute_bleu, compute_rouge
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
import re
from src.reddit.reddit import parse_answer
from function import filter_and_align_comments,create_infer_prompt
from src.anonymized.evaluate_anonymization import check_correctness
from src.models.model_factory import get_model
from src.configs.config import ModelConfig
from typing import Dict, Any,List
from llama_re import llm_response
import json
from accelerate.utils import infer_auto_device_map
from utils_rl import ModelDistributor
import time
# === 新增导入 ===
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
random.seed(42)
class TPMError(Exception):
    """TPM 不足导致的错误"""
    pass

class LLMTimeoutError(Exception):
    """LLM 响应超时"""
    pass


# === Basic Logging Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open("train_rl.json", "r", encoding="utf-8") as f:
    data_rl = json.load(f)

# === 初始化Accelerator和TensorBoard ===
accelerator = Accelerator()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/{timestamp}")

# === Config ===
# --- Model Configuration ---
MODEL_NAME = "/datadisk/LLaMA-Factory/models/llama-3.1-8b-instruct"
# 使用accelerator的设备设置
DEVICE = accelerator.device
logging.info(f"Using device: {DEVICE}")

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
logging.info(f"LoRA Config initialized: {lora_config}")


# --- GRPO Configuration ---
output_dir = "outputs_lora_b"
grpo_config = GRPOConfig(
    max_prompt_length=8192,
    max_completion_length=1024,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    logging_steps=10,
    num_generations=2,
    max_steps=200,
    save_steps=10,
    output_dir=output_dir,
    remove_unused_columns=False
)
logging.info(f"GRPO Config: {grpo_config}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created output directory: {output_dir}")

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True, 
        device_map="auto", 
        padding_side="right", 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
if tokenizer.pad_token is None:
    logging.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# === Load Model with LoRA ===
# base_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True,
#     device_map={"": DEVICE}
# )
# logging.info(f"Base model '{MODEL_NAME}' loaded.")


distributor = ModelDistributor(
    model_path=MODEL_NAME,  # 或本地路径
    cuda_list="0,1",  # 使用GPU 0和1
    memory_per_gpu="35GiB",
    torch_dtype=torch.float16
)


base_model = distributor.load_model()

base_model.enable_input_require_grads()

model = get_peft_model(base_model, lora_config)
logging.info("Applied LoRA adapter to the base model.")
model.print_trainable_parameters()

model.gradient_checkpointing_enable()

# === Dummy Attacker for Reward Simulation ===
class DummyPrivacyAttacker:
    def __init__(self, weights: List[float] = None):
        self.weights = weights or [0.8, 0.15, 0.05]
        self._model = None

    @property
    def model(self):
        if self._model is None:
            model_config = ModelConfig(
                name="deepseek-ai/DeepSeek-V3",
                provider="siliconflow",
                max_workers=4,
                args={"temperature": 0.1},
            )
            self._model = get_model(model_config)
        return self._model

    def attack(self, original: Dict[str, Any], protected: str) -> float:
        required_keys = {"pre", "pii_types", "pii_gt"}
        if not all(key in original for key in required_keys):
            raise ValueError(f"Missing keys in 'original'. Expected: {required_keys}")
        
        if len(original["pii_types"]) != len(original["pii_gt"]):
            raise ValueError("'pii_types' and 'pii_gt' length mismatch")
        flag=0
        infer_answer=[]
        try:
            anonymized = filter_and_align_comments(protected, original["pre"])
            prompt = create_infer_prompt(anonymized, original["pii_types"])
            for i in range(10):
                try:
                    input_ids_ = tokenizer.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(DEVICE)

                    # 2. Generate completions (protected text)
                    model.eval()
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=input_ids_,
                            max_new_tokens=1200,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=1
                        )

                    input_length = input_ids_.shape[1]
                    completions_tokens = output[:, input_length:]
                    completions = tokenizer.batch_decode(completions_tokens, skip_special_tokens=True)
                    result = completions[0]
                    break
                except Exception as e:
                    print(f"Error generating completion: {e}")
                    print("Retrying generation...")
                    time.sleep(30)     
            
            
            infer_answer = parse_answer(result, original["pii_types"])
        except Exception as e:
            print(f"Processing failed: {e}")
            flag=1

        if not infer_answer:
            flag=1
        
        score = 0.0
        bleu = compute_bleu(
                "\n".join([str(c) for c in original["pre"]]),
                "\n".join([str(c) for c in anonymized]),
            )
        rouge = compute_rouge(
                "\n".join([str(c) for c in original["pre"]]),
                ["\n".join([str(c) for c in anonymized])],
            )    
        rouge1=rouge[0]["rouge1"].fmeasure   
        rougeL= rouge[0]["rougeL"].fmeasure
        utility_score=(bleu+rouge1+rougeL)/3
        if flag==0:
            for pii, infer in infer_answer.items():
                try:
                    idx = original["pii_types"].index(pii)
                    gt = original["pii_gt"][idx]
                    is_correct = check_correctness(
                        gt, infer["guess"], infer["inference"], infer["inference"],pii, self.model, "model"
                    )
                    while len(is_correct)<3:
                        is_correct.append(0)
                    for i, weight in enumerate(self.weights):
                        if is_correct[i] == 0:
                            score += weight
                except Exception as e:
                    print(f"Skipping PII {pii}: {e}")
                    continue

        ano_score = score / len(infer_answer) if infer_answer else 0.0
        final_score=(utility_score+ano_score)/2
        if accelerator.is_main_process:
            logging.info(f"Final privacy score: {final_score:.2f}")
            
        return final_score
     
attacker = DummyPrivacyAttacker()

# === Reward Function ===
def compute_privacy_reward(prompts, completions, **kwargs):
    reward_datas=kwargs["reward_data"]
    rewards = []
    for prompt, completion in zip(reward_datas, completions):
        score = attacker.attack(prompt, completion)
        rewards.append(score)
    with open(f"scores/privacy_score_{timestamp}.txt", "a") as f:
        f.write(f"Final privacy score: {rewards}\n")    
    return rewards

# === Dataset ===
def build_dataset(Data):
    shuffled_data = list(Data)  # 转换为列表（如果还不是列表）
    random.shuffle(shuffled_data)
    return Dataset.from_list([
        {
            "input_text": item["input_text"],
            "reward_data": {
                "pre": item["reward_data"]["pre"],
                "pii_types": item["reward_data"]["pii_types"],
                "pii_gt": item["reward_data"]["pii_gt"]
            }
        }
        for item in shuffled_data
    ])

dataset = build_dataset(data_rl)

# === GRPO Trainer ===
trainer = GRPOTrainer(
    args=grpo_config,
    model=model,
    processing_class=tokenizer,
    reward_funcs=compute_privacy_reward
)
logging.info("GRPOTrainer initialized.")

# === 使用accelerator准备模型和优化器 ===
model, trainer = accelerator.prepare(model, trainer)


# # 检查哪些参数需要梯度
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")

# === Training Loop ===
global_step = 0
num_epochs = 3

def format_chat_prompt(text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text},{"role": "system", "content": "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."}],
        add_generation_prompt=True,
        tokenize=False,
    )

logging.info(f"Starting training for {num_epochs} epochs or max {grpo_config.max_steps} steps.")

for epoch in range(num_epochs):
    logging.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    if global_step >= grpo_config.max_steps:
        logging.info("Maximum steps reached. Stopping training.")
        break

    for index, example in enumerate(dataset):
        if global_step >= grpo_config.max_steps:
            break
            
        input_text = example["input_text"]
        reward_data = example["reward_data"]
        logging.info(f"\nProcessing Prompt {index+1}/{len(dataset)} (Step {global_step}): {example['input_text'][:80]}...")

        messages = [
            {"role": "system", "content": "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."},
            {"role": "user", "content": input_text}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        # 2. Generate completions (protected text)
        model.eval()
        all_completions_for_prompt = []
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=2
                )

            input_length = input_ids.shape[1]
            completions_tokens = output[:, input_length:]
            completions = tokenizer.batch_decode(completions_tokens, skip_special_tokens=True)
            all_completions_for_prompt.extend(completions)

            logging.debug(f"  Original: {input_text[:70]}...")
            for i, comp in enumerate(completions):
                logging.debug(f"  Generated {i+1}: {comp[:70]}...")

        except Exception as e:
            logging.error(f"Error during generation for prompt '{input_text[:50]}...': {e}")
            continue
        print(all_completions_for_prompt[0])
        # batch_data_for_trainer = [
        #     {"prompt": format_chat_prompt(input_text), "completion": c, "reward_data": reward_data}
        #     for c in all_completions_for_prompt
        # ]
        batch_data_for_trainer = []
        for completion in all_completions_for_prompt:
            # 对 Prompt 进行 Tokenization
            prompt_ids = tokenizer(
                format_chat_prompt(input_text),
                return_tensors="pt",  # 返回 PyTorch 张量
                padding=True,        # 自动填充
                truncation=True      # 自动截断
            ).input_ids.to(DEVICE)   # 移动到指定设备（如 GPU）

            # 对 Completion 进行 Tokenization
            completion_ids = tokenizer(
                completion,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(DEVICE)

            batch_data_for_trainer.append({
                "prompt": format_chat_prompt(input_text),          # Tokenized 后的 Prompt
                "completion": completion,  # Tokenized 后的 Completion
                "reward_data": reward_data         # 原始数据（用于计算 Reward）
            })
        #3. 计算奖励
        prompts=[item["prompt"] for item in batch_data_for_trainer]
        completions=[item["completion"] for item in batch_data_for_trainer]
        reward_data=[item["reward_data"] for item in batch_data_for_trainer]
        rewards = compute_privacy_reward(prompts, completions, reward_data=reward_data)

        # 使用 accelerator 准备数据
        batch_data_for_trainer = accelerator.prepare(batch_data_for_trainer)
        print("\n------------------------\n")

        logging.info(type(batch_data_for_trainer[0]["prompt"]))
        logging.info(type(batch_data_for_trainer[0]["completion"]))
        #logging.info(type(batch_data_for_trainer[0]["reward_data"]))
        #batch_data_for_trainer = accelerator.prepare(batch_data_for_trainer)
        logging.info(f"  Data prepared for training: {len(batch_data_for_trainer)} samples.")
        if batch_data_for_trainer:
            model.train()
  
            loss = trainer.training_step(model, batch_data_for_trainer, num_items_in_batch=len(batch_data_for_trainer))
            logging.info(f"  Step {global_step} complete")
            if loss is not None:
                # 记录loss到TensorBoard
                writer.add_scalar('Training/loss', loss.item(), global_step)
                logging.info(f"  Step {global_step} Loss: {loss.item():.4f}")
            
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    writer.add_scalar('Reward/avg_reward', avg_reward, global_step)
                    logging.info(f"  Step {global_step} reward: {avg_reward:.4f}")
                global_step += 1

            # 5. Save adapter checkpoint periodically
            if global_step % grpo_config.save_steps == 0:
                checkpoint_path = f"{grpo_config.output_dir}/checkpoint-{global_step}"
                try:
                    # 使用accelerator保存模型
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_path, save_function=accelerator.save)
                    logging.info(f"LoRA adapter saved to {checkpoint_path} at step {global_step}")
                except Exception as e:
                    logging.error(f"Error saving checkpoint at step {global_step}: {e}")

            # except Exception as e:
            #     logging.error(f"Error during training_step at global_step {global_step}: {e}")
        else:
            logging.warning(f"No completions generated for prompt {index+1}, skipping training step.")

# --- End of Training Loop ---
logging.info("Training loop finished.")

# Save final LoRA adapter
if global_step > 0:
    final_save_path = f"{grpo_config.output_dir}/final_adapter"
    try:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_save_path, save_function=accelerator.save)
        logging.info(f"Final LoRA adapter saved to {final_save_path}")
    except Exception as e:
        logging.error(f"Error saving final LoRA adapter: {e}")

# 关闭TensorBoard writer
writer.close()
print("Script finished.")