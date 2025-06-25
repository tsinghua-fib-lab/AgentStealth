import json
from datetime import datetime
    

# 使用示例

with open("good_levels_train_zi.json", "r", encoding="utf-8") as f:
    data=json.load(f)   
    
texts=[]      
with open("inference_5.jsonl","r",encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line))
llm_out=[]        
with open("ano_outputs_zi.jsonl","r",encoding="utf-8")  as f:
    for line in f:
        out=json.loads(line)
        llm_out.append(out)
with open("insights_570.json","r",encoding="utf-8") as f:    
    feature_insights=json.load(f)

ans=[]    
ans_rl=[]
system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
header1 = f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            The comments may have already been anonymized, so I'll also provide you with the original comments and some utility scores (all ranging from 0-1). (If the scores are > 0.9, it means that utility is well preserved, if the score is between 0.7and 0.9, it means the utility is acceptable ,if the scores are < 0.7, it means that utility is not well preserved).
            If you find the utility scores are not so high, maybe you should not change the comments too much. 
            Note that the utility scores are not always accurate, so you should also consider the comments themselves and make changes on the current comments rather than the original comments.
            
            

            """
header2=f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            
            

            """


footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."
for i in range(len(data)):
    id=data[i][3]
    piis=data[i][1]
    gt=data[i][2]
    per=[d for d in texts if d.get("username") == id][0]
    if len(data[i])==0:
        continue
    anon_level=data[i][0]
    for j in range(anon_level):
        comments_ori=per["comments"][0]["comments"]
        ori=[]
        for item in comments_ori:
            time_=item["timestamp"] if isinstance(item["timestamp"], datetime) else datetime.fromtimestamp(int(float(item["timestamp"])))
            ori.append(f"{time_.strftime('%Y-%m-%d')}: {item['text']}")
        comments_j=per["comments"][j]["comments"]
        com=[]
        for item in comments_j:
            time_=item["timestamp"] if isinstance(item["timestamp"], datetime) else datetime.fromtimestamp(int(float(item["timestamp"])))
            com.append(f"{time_.strftime('%Y-%m-%d')}: {item['text']}")
        Ori="\n".join(ori)
        Com="\n".join(com)
        previous_inferences = per["comments"][j]["predictions"]["deepseek-ai/DeepSeek-V3"]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]
        pii_insights=""   
        for pii in piis:

            insights=feature_insights.get(pii, {}).get("insights", [])[:3]
            if len(insights) >0:
                insight="\n".join(insights)
                pii_insights += f"\nFor {pii}:\nThe insights for anonymization:\n{insight}\n"
            else:
                pii_insights += f"\nFor {pii}:\nNo insights available.\n"     
        res={}         
        model_utility =per["comments"][j]["utility"]
        if model_utility:
            key = next(iter(model_utility))
            model_utility=model_utility[key]
        if "bleu" in model_utility:
            res["bleu"] = model_utility["bleu"]
        if "rouge" in model_utility:
            res["rouge1"] = model_utility["rouge"][0]["rouge1"][2]
            res["rougeL"] = model_utility["rouge"][0]["rougeL"][2]
        utility=f"bleu: {res.get('bleu', 1)}\nrouge1: {res.get('rouge1', 1)}\nrougeL: {res.get('rougeL', 1)}\n" 
        average_utility = (res.get('bleu', 1)+res.get('rouge1', 1)+res.get('rougeL', 1))/3
        if j<2:
            pre_utility=1
        else:    
            Res={}
            model_utility = per["comments"][j-1]["utility"]
            if model_utility:
                key = next(iter(model_utility))
                model_utility=model_utility[key]
            if "bleu" in model_utility:
                Res["bleu"] = model_utility["bleu"]
            if "rouge" in model_utility:
                Res["rouge1"] = model_utility["rouge"][0]["rouge1"][2]
                Res["rougeL"] = model_utility["rouge"][0]["rougeL"][2]   
            pre_utility = (Res.get('bleu', 1)+Res.get('rouge1', 1)+Res.get('rougeL', 1))/3            
        intermediate = f"\n\nOriginal comments:\n{Ori}\n\nCurrent comments:\n{Com}\n \nInferences:\n\n{inference_string}\n Utility scores:\n\n{utility}\n\nInsights:\n\n{pii_insights}"
        ano_out=[d for d in llm_out if d.get("username") == id and d.get("length",-1)==j][0]
        if (pre_utility-average_utility)>0.075:
            header=header1
            intermediate = f"\n\nOriginal comments:\n{Ori}\n\nCurrent comments:\n{Com}\n \nInferences:\n\n{inference_string}\n Utility scores:\n\n{utility}\n\nInsights:\n\n{pii_insights}"
        else:
            header=header2
            intermediate = f"\n\nCurrent comments:\n{Com}\n \nInferences:\n\n{inference_string}\nInsights:\n\n{pii_insights}"
        input_text = f"{header}\n{intermediate}\n{footer}"
        input_text_= f"{header}\n{footer}\n{intermediate}"
        output_text = ano_out["answer"]
        new_data = {
                "instruction": input_text,
                "input": "",
                "output": output_text,
                "system": system_prompt
            }
        rl_data={"input_text": input_text,
                 "reward_data":{
                     "pre":com,
                     "pii_types":piis,
                     "pii_gt":gt
                 }

        }
        ans.append(new_data)
        ans_rl.append(rl_data)
print(len(ans))
with open("train_anonymization_zi.json","w",encoding="utf-8") as f:
    json.dump(ans, f, ensure_ascii=False, indent=4)
with open("train_rl_zi.json","w",encoding="utf-8") as f:
    json.dump(ans_rl, f, ensure_ascii=False, indent=4)