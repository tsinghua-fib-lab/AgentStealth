import requests
from typing import List, Dict, Tuple, Iterator, Optional, Any
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.configs import ModelConfig
from src.prompts import Prompt, Conversation
from .model import BaseModel
from openai import RateLimitError
from requests.exceptions import RequestException
from credentials import siliconflow_api_key
class SiliconFlowModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        
        # 设置API默认参数
        self.api_url =  "https://api.siliconflow.cn/v1/chat/completions"
        self.api_key =  siliconflow_api_key
        
        # 模型参数配置
        self.model_name = self.config.name
        self.temperature = self.config.args.get("temperature", 0.1)
        self.max_tokens = self.config.args.get("max_tokens", 2000)
        self.top_p = self.config.args.get("top_p", 0.7)
        self.top_k = self.config.args.get("top_k", 50)
        self.frequency_penalty = self.config.args.get("frequency_penalty", 0.5)
        self.n = self.config.args.get("n", 1)
        self.stop_strs = self.config.args.get("stop_strs", None)

    def _predict_call(self, input: List[Dict[str, str]]) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": input,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
            "stop": self.stop_strs,
            "response_format": {"type": "text"}
        }

        for _ in range(3):  # 重试机制
            try:
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    time.sleep(30)
                else:
                    raise
            except Exception as e:
                print(f"API Error: {str(e)}")
                time.sleep(20)
        raise RuntimeError("API Error: Max retries exceeded")

    def predict(self, input: Prompt, **kwargs) -> str:

        messages = []
        
        # 系统提示设置
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are an expert investigator with years of experience in text analysis."
            })
            
        # 用户输入处理
        processed_input = self.apply_model_template(input.get_prompt())
        messages.append({"role": "user", "content": processed_input})
        
        return self._predict_call(messages)

    def predict_string(self, input: str, **kwargs) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input}
        ]
        return self._predict_call(messages)

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 4
        base_timeout = kwargs["timeout"] if "timeout" in kwargs else 240

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
                results = executor.map(
                    lambda id: (id, inputs[id], self.predict(inputs[id])),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Profiles",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield (orig, answer)
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except RateLimitError as r:
                    print(f"Rate_limit {r}")
                    time.sleep(30)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(600, timeout)
                retry_ctr += 1

        # return answered_prompts

    def continue_conversation(self, input: Conversation, **kwargs) -> str:

        messages = []
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
            
        for message in input.prompts:
            messages.append({
                "role": message.role,
                "content": self.apply_model_template(message.get_prompt())
            })
            
        return self._predict_call(messages)