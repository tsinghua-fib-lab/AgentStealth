
from src.configs.config import ModelConfig
from src.models.model import BaseModel
from src.models.model_factory import get_model
import json
from itertools import islice
import argparse
from typing import List,Union
import os
import time as TIME
import random
import re
from typing import List, Dict, Optional
from datetime import datetime
def type_to_str(pii_type: str) -> str:
    if pii_type == "income":
        pii_type_str = "yearly income"
    elif pii_type == "education":
        pii_type_str = "level of education"
    elif pii_type == "pobp":
        pii_type_str = "place of birth"
    elif pii_type == "location":
        pii_type_str = "current place of living"
    elif pii_type == "married":
        pii_type_str = "relationship status"
    else:
        pii_type_str = pii_type
    return pii_type_str
def process_llm_output(llm_output: str, previous_insight: Dict) -> Dict:
    """处理LLM输出并返回优化后的洞察规则"""
    #print(llm_output)
    # 初始化数据结构
    rules = [
        {"text": insight, "score": score}
        for insight, score in zip(
            previous_insight.get("insights", []),
            previous_insight.get("scores", [])
        )
    ]

    # 处理每行输出
    for line in llm_output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # 解析操作
        op, rule_num, content = parse_operation(line)
        if not op:
            continue

        # 执行操作
        try:
            if op == "ADD":
                rules.append({"text": content, "score": 2})
            
            elif op == "EDIT" and 0 <= rule_num < len(rules):
                rules[rule_num]["text"] = content
                rules[rule_num]["score"] += 1
                
            elif op == "UPVOTE" and 0 <= rule_num < len(rules):
                rules[rule_num]["score"] += 1
                
            elif op == "DOWNVOTE" and 0 <= rule_num < len(rules):
                rules[rule_num]["score"] -= 1
                
        except (IndexError, TypeError) as e:
            print(f"Error processing line: {line}\nError: {e}")

    # 过滤和限制规则
    rules = [r for r in rules if r["score"] > 0]
    rules.sort(key=lambda x: -x["score"])  # 按分数降序
    
    return {
        "insights": [r["text"] for r in rules[:5]],
        "scores": [r["score"] for r in rules[:5]]
    }

def parse_operation(line: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """解析操作类型、规则编号和内容"""
    match = re.match(r"^(ADD|EDIT|UPVOTE|DOWNVOTE)<(\d+)>:\s*(.+)$", line)
    if match:
        return match.group(1), int(match.group(2)) - 1, match.group(3)
    return None, None, None

parser = argparse.ArgumentParser(description="config")
parser.add_argument('--input_path', type=str, help="input path of evals")
parser.add_argument('--output_path', type=str, help="output path")
parser.add_argument('--input_path_comment', type=str, help="input path of comments")
parser.add_argument('--offset', type=str, default='0', help="offset of the first eval")

parser.add_argument(
        "--inference_model_to_eval",
        type=str,
        default="deepseek-ai/DeepSeek-V3",
        help="Model to evaluate",
    )
args=parser.parse_args()
model_config = ModelConfig(
        name="deepseek-ai/DeepSeek-V3",
        provider="siliconflow",
        max_workers=4,
        args={
            "temperature": 0.1,
        },
    )
model = get_model(model_config)
offset=int(args.offset)
inference_model_norm = args.inference_model_to_eval.replace("/", "_")


with open(os.path.join(args.input_path, f"eval_{inference_model_norm}_out_{offset}.jsonl"), "r", encoding="utf-8") as f:
    data_list  = json.load(f)

data_comments = []
with open(os.path.join(args.input_path_comment, "inference_5.jsonl"), "r", encoding="utf-8") as f:
    # 跳过前offset行，读取接下来的count行
    for line in islice(f, offset, offset + 10):
        try:
            data_comments.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 跳过无效JSON行
output_file=os.path.join(args.output_path, f"insights_{offset}.json")
previous_insights = {}
if offset>9:
    previous_offset = offset - 10
    previous_insight_file = os.path.join(args.output_path, f"insights_{previous_offset}.json")
    if os.path.exists(previous_insight_file):
        with open(previous_insight_file, "r", encoding="utf-8") as f:
            previous_insights = json.load(f)
        print(f"Loading previous insights from {previous_insight_file}")    


infer_examples={}
for i in range(len(data_list)):
    if "0" not in data_list[i]:
        continue
    else:    
        base=data_list[i]["0"].get("is_correct",[0,0,0])
    while len(base) < 3:
        base.append(0)
    if base==[0,0,0]:
        continue    
    pii=data_list[i]["pii_type"]
    if not pii:
        continue
    username=data_list[i]["id"]
    if pii not in infer_examples:
        infer_examples[pii]={"success": [], "pairs": []}
    pairs=[]    
    per=[d for d in data_comments if d.get("username") == username][0]
    for j in range(5):
        if str(j+1) not in data_list[i]:
            continue
        correct=data_list[i][str(j+1)].get("is_correct",[1,1,1])
        success=True
        while len(correct) < 3:
            correct.append(0)
        dif=[0,0,0]    
        for k in range(3):
            dif[k]=correct[k]-base[k]
        if correct[0]==1:
            success=False 
        else:       
            if dif[0] <0:
                success=True
            elif dif[0] >0:
                success=False
            else:
                if dif[1] <0:
                    success=True
                elif dif[1] >0:
                    success=False
                else:
                    if dif[2] <0:
                        success=True
                    else:
                        success=False       
        infer=""            
        infer += f"Type: {type_to_str(pii)}\n"
        infer += f"Inference: {data_list[i][str(j+1)]['inference']}\n"
        infer += f"Guess: {data_list[i][str(j+1)]['pred']}\n"    
        ori_infer=""            
        ori_infer += f"Type: {type_to_str(pii)}\n"
        ori_infer += f"Inference: {data_list[i]['0']['inference']}\n"
        ori_infer += f"Guess: {data_list[i]['0']['pred']}\n"

        comments_ori=per["comments"][0]["comments"]
        ori=[]
        for item in comments_ori:
            time_=item["timestamp"] if isinstance(item["timestamp"], datetime) else datetime.fromtimestamp(int(float(item["timestamp"])))
            ori.append(f"{time_.strftime('%Y-%m-%d')}: {item['text']}")
        comments_j=per["comments"][j+1]["comments"]
        com=[]
        for item in comments_j:
            time_=item["timestamp"] if isinstance(item["timestamp"], datetime) else datetime.fromtimestamp(int(float(item["timestamp"])))
            com.append(f"{time_.strftime('%Y-%m-%d')}: {item['text']}")
        Ori="\n".join(ori)
        Com="\n".join(com)
        if success:
            infer_examples[pii]["success"].append(f"Original Comments:\n{Ori}\n\nAnonymized Comments:\n{Com}\n\nInference:\n{ori_infer}")
            if len(pairs) >0:
                for d in range(len(pairs)):
                    pairs[d]= pairs[d]+f"Success_anonymized Comments:\n{Com}\n"
                infer_examples[pii]["pairs"].extend(pairs)
            break
        else:
            pairs.append(f"Original Comments:\n{Ori}\nInference:\n{ori_infer}\n Failure_anonymized Comments:\n{Com}\n The Failure is because that the pii still can be infered:\n{infer}\n")

feature_insights = previous_insights

for feature, examples in infer_examples.items():
    # Prepare optimized prompt with response samples
    success_text = "\n".join(examples["success"])
    pairs_batches = examples["pairs"]
    print(f"Analyzing {feature}...")
    print(len(examples["success"]))
    print(len(pairs_batches))
    previous_insight = previous_insights.get(feature, {}).get('insights', []) if offset > 0 else []
    scores=previous_insights.get(feature, {}).get('scores', []) if offset > 0 else []
    #print(previous_insight)
    system="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories.  "
    success_prompt = f"""You will be given successful tasks trials in which you anonymize the original texts from the inferneces.
    Here are the trials:\n{success_text}\n
    Here are the EXISTING RULES:\n

    """
    if len(previous_insight) > 0:   
        for i, insight in enumerate(previous_insight):
            success_prompt += f"{i+1}. {insight}\n"
        success_prompt += "\n"
    else:  
        success_prompt +="There are no existing rules.\n\n"  
    success_prompt += """By examining successful trials ,and the list of existing rules, you can perform the following operations: add, edit, downvote, or upvote so that the new rules are GENERAL and HIGH LEVEL insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought.
    Follow the below format:

<OPERATION><RULE NUMBER>:<RULE>

The available operations are: 
UPVOTE(if the existing rule is strongly relevant for the task),
DOWNVOTE(if one existing rule is contradictory or similar/duplicated to other existing rules), 
EDIT(if any existing rule is not general enough or can be enhanced,rewrite and improve it), 
ADD(add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below:

UPVOTE<EXISTING RULE NUMBER>:<EXISTING RULE>

DOWNVOTE<EXISTING RULE NUMBER>:<EXISTING RULE>

EDIT<EXISTING RULE NUMBER>:<NEW MODIFIED RULE>

ADD<NEW RULE NUMBER>:<NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times.
Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Note that every insight you add or edit must be less than 100 words.
Below are the operations you do to the above list of EXISTING RULES:\n\n"""
    

    while True:
        try:
            print(success_prompt)
            result = model._predict_call([{"role":"system", "content": system}, {"role":"user", "content": success_prompt}],model='deepseek-ai/DeepSeek-V3')['choices'][0]['message']['content']
            new_insight=process_llm_output(result, {"insights": previous_insight, "scores": scores})
            #print(new_insight)        
            break
        except Exception as e:
            print(f"Error summarizing {feature} success: {e}")
            print("Retrying...")
            TIME.sleep(30)
    for pair in pairs_batches:
        system="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories.  "
        pair_prompt = f"""You will be given two previous tasks trials in which you anonymize the original texts from the inferneces. One success and one failure for you to compare and critique.
        Here are the trials:\n{pair}\n
        Here are the EXISTING RULES:\n

        """
        if len(new_insight["insights"]) > 0:   

            for i, insight in enumerate(new_insight["insights"]):
                pair_prompt += f"{i+1}. {insight}\n"
            pair_prompt += "\n"
        else:  
            pair_prompt +="There are no existing rules.\n\n"  
        pair_prompt += """By examining and contrasting the successful trial,and the list of existing rules, you can perform the following operations: add, edit, downvote, or upvote so that the new rules are GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought.
        Follow the below format:

<OPERATION><RULE NUMBER>:<RULE>

The available operations are: 
UPVOTE(if the existing rule is strongly relevant for the task),
DOWNVOTE(if one existing rule is contradictory or similar/duplicated to other existing rules), 
EDIT(if any existing rule is not general enough or can be enhanced,rewrite and improve it), 
ADD(add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below:

UPVOTE<EXISTING RULE NUMBER>:<EXISTING RULE>

DOWNVOTE<EXISTING RULE NUMBER>:<EXISTING RULE>

EDIT<EXISTING RULE NUMBER>:<NEW MODIFIED RULE>

### ADD<NEW RULE NUMBER>:<NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times.
Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Note that every insight you add or edit must be less than 100 words.
Below are the operations you do to the above list of EXISTING RULES:\n\n"""
        

        while True:
            try:
                print(pair_prompt)
                result = model._predict_call([{"role":"system", "content": system}, {"role":"user", "content": pair_prompt}],model='deepseek-ai/DeepSeek-V3')['choices'][0]['message']['content']
                new_insight=process_llm_output(result, new_insight)
                #print(new_insight)        
                break
            except Exception as e:
                print(f"Error summarizing {feature} pairs: {e}")
                print("Retrying...")
                TIME.sleep(30)
        

    feature_insights[feature] = new_insight
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(feature_insights, f, ensure_ascii=False, indent=4)       
print("Summary done for offset:",offset)

