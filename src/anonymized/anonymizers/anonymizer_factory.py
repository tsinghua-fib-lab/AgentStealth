from .azure_anonymizer import AzureAnonymizer
from .llm_anonymizers import LLMFullAnonymizer, LLMBaselineAnonymizer, LLMBaselineAnonymizer2, LLMBaselineAnonymizer3,LLMSummaryAnonymizer,LLMStyleAnonymizer,LLMCodingAnonymizer,LLMCodingAnonymizer1,LLMCodingAnonymizer2
from .anonymizer import Anonymizer

from src.configs import AnonymizationConfig
from src.models.model_factory import get_model

def get_anonymizer(cfg: AnonymizationConfig) -> Anonymizer:
    
    if cfg.anonymizer.anon_type == "azure":
        return AzureAnonymizer(cfg.anonymizer)
    elif cfg.anonymizer.anon_type == "llm":
        model = get_model(cfg.anon_model)
        return LLMFullAnonymizer(cfg.anonymizer, model)
    elif cfg.anonymizer.anon_type == "llm_base":
        model = get_model(cfg.anon_model)
        return LLMBaselineAnonymizer(cfg.anonymizer, model)
    elif cfg.anonymizer.anon_type == "llm_base2":
        model = get_model(cfg.anon_model)
        return LLMBaselineAnonymizer2(cfg.anonymizer, model)
    elif cfg.anonymizer.anon_type == "llm_base3":
        model = get_model(cfg.anon_model)
        return LLMBaselineAnonymizer3(cfg.anonymizer, model) 
    elif cfg.anonymizer.anon_type == "llm_summary":
        model = get_model(cfg.anon_model)
        return LLMSummaryAnonymizer(cfg.anonymizer, model) 
    elif cfg.anonymizer.anon_type == "llm_style":
        model = get_model(cfg.anon_model)
        return LLMStyleAnonymizer(cfg.anonymizer, model)   
    elif cfg.anonymizer.anon_type == "llm_coding":
        model = get_model(cfg.anon_model)
        return LLMCodingAnonymizer(cfg.anonymizer, model)  
    elif cfg.anonymizer.anon_type == "llm_coding1":
        model = get_model(cfg.anon_model)
        return LLMCodingAnonymizer1(cfg.anonymizer, model)         
    elif cfg.anonymizer.anon_type == "llm_coding2":
        model = get_model(cfg.anon_model)
        return LLMCodingAnonymizer2(cfg.anonymizer, model)          
    else:
        raise ValueError(f"Unknown anonymizer type {cfg.anonymizer.anon_type}")