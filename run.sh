
CUDA_VISIBLE_DEVICES=1 python main.py --config_path configs/anonymization/reddit_coding_test.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config_path configs/anonymization/reddit_coding_test_eval.yaml --level all
CUDA_VISIBLE_DEVICES=1 python src/anonymized/evaluate_anonymization.py --in_path anonymized_results/coding_test/eval_inference_results.jsonl --decider "model" --out_path anonymized_results/coding_test/eval --score --eval_level 0,5



