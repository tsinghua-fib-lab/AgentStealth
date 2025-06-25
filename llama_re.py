import requests 
from typing import Any, Dict, List, Optional
import json

url = "https://api.siliconflow.cn/v1/chat/completions"






def llm_response(prompt: str,model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B' ,temperature: float = 0.1,stop_strs = None,n=1,max_tokens=1000) :
    payload = {
        "model": model,
        "messages": prompt,
        "stream": False,
        "max_tokens": max_tokens,
        "stop": stop_strs,
        "temperature": temperature,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": n,
        "response_format": {"type": "text"}
    }
    headers = {
        "Authorization": "Bearer sk-hvvzcjtdefwzmrnifvyrvisktrfssstmbkwnkzkxncnydvhv",
        "Content-Type": "application/json"
    }

    
    response = requests.request("POST", url, json=payload, headers=headers)

    return json.loads(response.text)


    
    
# print(llm_response([{"role": "user", "content": "你好，请问有什么可以帮助您的吗？"}]))
# 查看第一行的具体值




