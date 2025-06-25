import yaml
import argparse
parser = argparse.ArgumentParser(description="config")
parser.add_argument('--file', type=str, help=" path of evals")
parser.add_argument('--offset', type=str, default=0, help="offset of the first eval")

args=parser.parse_args()
offset=int(args.offset)
with open(args.file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
if 'task_config' in config :
    config['task_config']['offset'] = offset+10
    config['task_config']['anonymizer']["summary_path"] = f"summary/coding4/insights_{offset}.json"
else:
    raise ValueError("YAML文件中没有找到task_config.offset配置项")
    
    # 写回文件
with open(args.file, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)       