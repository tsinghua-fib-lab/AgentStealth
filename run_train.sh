#!/bin/bash

#!/bin/bash


for i in $(seq 0 57); do
    offset=$((i*10))
    echo "======= Start offset=$offset (i=$i) =======" 

    CUDA_VISIBLE_DEVICES=1 python main.py --config_path configs/anonymization/reddit_coding_train.yaml || exit 1
    
    # 修正2：使用 $(( )) 进行算术运算
    CUDA_VISIBLE_DEVICES=1 python src/anonymized/evaluate_anonymization.py \
        --in_path anonymized_results/coding_train/inference_5.jsonl \
        --decider "model" \
        --out_path anonymized_results/coding_train/eval \
        --score \
        --offset $offset || exit 1
    
    CUDA_VISIBLE_DEVICES=1 python summary_coding.py \
        --input_path anonymized_results/coding_train/eval \
        --input_path_comment anonymized_results/coding_train \
        --output_path summary/coding_train \
        --offset $offset || exit 1
    
    python change_config.py \
        --file configs/anonymization/reddit_coding_train.yaml \
        --offset $offset || exit 1
    echo "--> Finish:offset=$offset"  # 每轮结束后输出
done        
#python src/anonymized/evaluate_anonymization.py --in_path anonymized_results/deepseek/eval/inference_0_100.jsonl --decider "model" --out_path anonymized_results/summary_step/synthPAI/eval --score --eval_level 0

#python main.py --config_path configs/anonymization/summary_step/reddit_deepseek_PAI_1.yaml
#python src/anonymized/evaluate_anonymization.py --in_path anonymized_results/summary_step/synthPAI/inference_1.jsonl --decider "model" --out_path anonymized_results/summary_step/synthPAI/eval --score --eval_level 1
#python summary_infer.py --input_path anonymized_results/summary_step/synthPAI/eval --input_path_comment anonymized_results/summary_step/synthPAI --output_file summary/summary_step/synthPAI/summary_1.jsonl --level 1 

