from src.configs import ModelConfig

from .model import BaseModel
from .open_ai import OpenAIGPT
from .siliconflow import SiliconFlowModel
from .open_ai_slm import OpenAISLM
def get_model(config: ModelConfig) -> BaseModel:
    if config.provider == "openai" or config.provider == "azure":
        return OpenAIGPT(config)
    elif config.provider == "siliconflow":
        return SiliconFlowModel(config)   
    elif config.provider == "openai_slm":
        return OpenAISLM(config)     
    elif config.provider == "loc":    
        raise NotImplementedError

    else:
        raise NotImplementedError
