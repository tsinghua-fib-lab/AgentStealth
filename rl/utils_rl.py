import torch
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.bitsandbytes import set_module_quantized_tensor_to_device
import os
from tqdm import tqdm

class ModelDistributor:
    def __init__(self, model_path, cuda_list='0,1', memory_per_gpu='10GiB', torch_dtype=torch.float16):
        """
        初始化分布式模型部署器
        
        参数:
            model_path: 模型路径 (本地或HuggingFace模型名)
            cuda_list: 逗号分隔的GPU ID列表 (如 '0,1,2')
            memory_per_gpu: 每个GPU的内存分配 (如 '10GiB')
            torch_dtype: 模型精度 (torch.float16/torch.bfloat16)
        """
        self.model_path = model_path
        self.cuda_list = [int(x) for x in cuda_list.split(',') if x.strip()]
        self.max_memory = {gpu_id: memory_per_gpu for gpu_id in self.cuda_list}
        self.torch_dtype = torch_dtype
        self.model = None
        self.device_map = None

    def load_model(self):
        """加载模型并自动分配到多GPU"""
        # 1. 获取模型配置
        config = AutoConfig.from_pretrained(self.model_path)
        
        # 2. 获取模型的不可分割模块列表
        with init_empty_weights():
            temp_model = AutoModelForCausalLM.from_config(config)
            self.no_split_modules = temp_model._no_split_modules
            del temp_model
        
        print(f"不可分割模块: {self.no_split_modules}")

        # 3. 初始化空模型 (meta设备)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=self.torch_dtype
            )

        # 4. 自动推断设备映射
        self.device_map = infer_auto_device_map(
            self.model,
            max_memory=self.max_memory,
            no_split_module_classes=self.no_split_modules,
            dtype=self.torch_dtype
        )
        print("设备映射方案:", self.device_map)

        # 5. 加载检查点并分配到设备
        self._load_checkpoints()
        
        # 6. 分发模型
        self.model = dispatch_model(self.model, device_map=self.device_map)
        
        return self.model

    def _load_checkpoints(self):
        """权重加载的兼容性实现"""
        # 尝试从不同位置导入load_checkpoint_and_dispatch
        for import_path in [
            'accelerate.utils',
            'accelerate',
            'accelerate.checkpointing'
        ]:
            try:
                module = __import__(import_path, fromlist=['load_checkpoint_and_dispatch'])
                loader = getattr(module, 'load_checkpoint_and_dispatch', None)
                if loader:
                    self.model = loader(
                        self.model,
                        self.model_path,
                        device_map=self.device_map,
                        dtype=self.torch_dtype
                    )
                    print("使用accelerate优化加载")
                    return
            except (ImportError, AttributeError):
                continue
        
        # 如果所有导入尝试都失败，使用手动加载
        print("未找到accelerate的优化加载方法，使用手动加载")
        self._manual_load_checkpoints()

    def _manual_load_checkpoints(self):
        """手动实现权重加载"""
        try:
            from safetensors.torch import load_file as safe_load
            use_safe = True
        except ImportError:
            use_safe = False
        
        # 确定检查点文件
        checkpoint_files = []
        for fname in os.listdir(self.model_path):
            if fname.endswith(".safetensors") and use_safe:
                checkpoint_files.append(fname)
            elif fname.endswith(".bin"):
                checkpoint_files.append(fname)
        
        if not checkpoint_files:
            raise FileNotFoundError(f"在 {self.model_path} 中未找到模型权重文件 (.bin/.safetensors)")

        # 加载每个权重文件
        for file in tqdm(checkpoint_files, desc="加载权重文件"):
            file_path = os.path.join(self.model_path, file)
            if file.endswith(".safetensors"):
                state_dict = safe_load(file_path)
            else:
                state_dict = torch.load(file_path, map_location="cpu")
            
            for name, tensor in state_dict.items():
                # 找到对应的模块
                module = self.model
                parts = name.split('.')
                for part in parts[:-1]:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                
                # 获取目标设备
                device = self.device_map.get(name, f"cuda:{self.cuda_list[0]}")
                if isinstance(device, str) and device.startswith("cuda:"):
                    device = torch.device(device)
                
                # 设置权重
                if hasattr(module, '_hf_hook'):
                    # 处理accelerate的hook
                    module._hf_hook.pre_forward(module)
                set_module_quantized_tensor_to_device(
                    module,
                    parts[-1],
                    device,
                    value=tensor,
                    dtype=self.torch_dtype
                )

    def print_device_map_summary(self):
        """打印设备分配统计信息"""
        device_stats = {}
        for layer, device in self.device_map.items():
            if isinstance(device, torch.device):
                dev_str = str(device)
            else:
                dev_str = device
            device_stats[dev_str] = device_stats.get(dev_str, 0) + 1
        
        print("\n设备分配统计:")
        for dev, count in device_stats.items():
            print(f"{dev}: {count} layers")

# 使用示例
if __name__ == "__main__":
    # 初始化部署器
    distributor = ModelDistributor(
        model_path="your_model_path",  # 替换为实际路径
        cuda_list="0,1",  # 使用GPU 0和1
        memory_per_gpu="10GiB",
        torch_dtype=torch.float16
    )
    
    # 加载模型
    model = distributor.load_model()
    distributor.print_device_map_summary()
    
    # 使用模型
    input_ids = torch.tensor([[1, 2, 3]]).to(f"cuda:{distributor.cuda_list[0]}")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50)
    print("生成结果:", outputs)