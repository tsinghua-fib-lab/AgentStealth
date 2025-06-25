import json
def find_best_anon_level(data_list):
    l=len(data_list)
    total_score=[0,0,0,0,0]
    gt=[]
    pii=[]
    id=data_list[0]["id"]
    for data in data_list:
        gt.append(data["gt"])
        pii.append(data["pii_type"])
        base=data["0"].get("is_correct",[0,0,0])
        while len(base) < 3:
            base.append(0)
        for anon_level in range(1,6):  # 假设 anon_level 范围是 0-5
            key = str(anon_level)
            if key not in data:
                continue  # 跳过不存在的 anon_level

            entry = data[key]

            # 条件1：is_correct 第一项不为 1
            
            is_correct = entry.get("is_correct", [0,0,0])
            while len(is_correct) < 3:
                is_correct.append(0)
            
            dif=[0,0,0]    
            for k in range(3):
                dif[k]=is_correct[k]-base[k]
            if is_correct[0]==1:
                score_1=0 
            else:       
                if dif[0] <0:
                    score_1=1
                elif dif[0] >0:
                    score_1=0
                else:
                    if dif[1] <0:
                        score_1=1
                    elif dif[1] >0:
                        score_1=0
                    else:
                        if dif[2] <0:
                            score_1=1
                        else:
                            score_1=0       
                
            # 不符合条件

            # 检查是否有 utility 数据
            utility = entry.get("utility", {})
            if not utility:
                score_2=0
            else:      # 如果没有 utility 数据则跳过

            # 计算综合得分（权重可调整）
                score_2 = (
                    1/6 * utility.get("bleu", 0) +
                    1/6 * utility.get("rouge1", 0) +
                    1/6 * utility.get("rougeL", 0) +
                    1/60 * utility.get("deepseek-ai/DeepSeek-V3_readability", 0) +
                    1/60 * utility.get("deepseek-ai/DeepSeek-V3_meaning", 0) +
                    1/6 * (utility.get("deepseek-ai/DeepSeek-V3_hallucination", 0) )
                )
            total_score[anon_level-1]+=score_1*score_2    
    total_score=[number/l for number in total_score]
    max_value = max(total_score)

    # 找到最大值的索引
    max_index = total_score.index(max_value)   
    if max_value > 0:
        return [max_index+1,pii,gt,id,max_value]  
    else:
        return []   


def find_good_anon_levels(data, threshold=0.7):

    good_levels = []
    base=data["0"].get("is_correct",[0,0,0])
    for anon_level in range(1,6):  # 假设 anon_level 范围是 0-5
        key = str(anon_level)
        if key not in data:
            continue  # 跳过不存在的 anon_level

        entry = data[key]

        # 条件1：is_correct 第一项不为 1
        
        is_correct = entry.get("is_correct", [0,0,0])
        while len(is_correct) < 3:
            is_correct.append(0)
        if is_correct[0]==1:
            continue
        elif is_correct[1]==1 and base==[0,1,0]:
            continue
        elif is_correct[2]==1 and base==[0,0,1]:
            continue    
        
          # 不符合条件

        # 检查是否有 utility 数据
        utility = entry.get("utility", {})
        if not utility:
            continue  # 如果没有 utility 数据则跳过

        # 计算综合得分（权重可调整）
        score = (
            1/6 * utility.get("bleu", 0) +
            1/6 * utility.get("rouge1", 0) +
            1/6 * utility.get("rougeL", 0) +
            1/60 * utility.get("deepseek-ai/DeepSeek-V3_readability", 0) +
            1/60 * utility.get("deepseek-ai/DeepSeek-V3_meaning", 0) +
            1/6 * (utility.get("deepseek-ai/DeepSeek-V3_hallucination", 0) )
        )

        # 如果得分 >= 阈值，则加入结果列表
        if score >= threshold:
            good_levels.append((anon_level, data["pii_type"],data["gt"],score))
    
    # 按得分从高到低排序
    good_levels.sort(key=lambda x: x[3], reverse=True)
    
    return good_levels
data_list=[]
for i in range(0,58):
    offset=10*i
    with open(f"eval4\eval_deepseek-ai_DeepSeek-V3_out_{offset}.jsonl", "r", encoding="utf-8") as f:
        d = json.load(f)
    data_list.extend(d)     
ans=[]    
num=0
id=data_list[0]["id"]
datalist=[]
# 运行分析（默认阈值设为 0.7）
num_=0
cnt=0
for data in data_list:
    if '0' not in data:
        continue
    if data["id"]!=id:
        print(id)
        good_level=find_best_anon_level(datalist)
        num_+=1
        if good_level:
            num+=1
            ans.append(good_level)
            cnt=cnt+good_level[0]
        id=data["id"]
        datalist=[data]
    else:
        datalist.append(data)

# 打印结果
print(num)
print(num_)
print(cnt)
with open("good_levels_train_zi.json", "w", encoding="utf-8") as f:
    json.dump(ans, f, ensure_ascii=False, indent=4)

# with open("eval_deepseek-ai_DeepSeek-V3_out.jsonl", "r", encoding="utf-8") as f:
#     data_list = json.load(f) 
# ans=[]    
# num=0
# # 运行分析（默认阈值设为 0.7）
# for data in data_list:
#     good_levels = find_good_anon_levels(data, threshold=0.75)
#     if good_levels:
#         num+=1
#     ans.append(good_levels)
# # 打印结果
# print(num)
# with open("good_levels.json", "w", encoding="utf-8") as f:
#     json.dump(ans, f, ensure_ascii=False, indent=4)
