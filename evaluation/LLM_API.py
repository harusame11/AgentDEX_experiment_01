import os
from tavily import TavilyClient
from openai import OpenAI
from zai import ZhipuAiClient

os.environ["DEEPSEEK_API_KEY"] = "sk-20a882166b5041e4bc34af189d2b993a" 
os.environ["TAVILY_API_KEY"] = "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"
os.environ["SILICON_API_KEY"] = "sk-khlysjcpkbdjtkqmyxtmktiqpowxogkkssqhzhotkhknzpcy"
os.environ["OPENROUTER_API_KEY"] = "" 
ds_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)
client = ZhipuAiClient(api_key="your-api-key")  # 请填写您自己的 API Key
silicon_client = OpenAI(
    api_key=os.environ.get("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)
openrouter_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
) 
local_client = OpenAI(
    base_url="http://192.168.2.238:8000/v1", 
    api_key="EMPTY" 
)

CLIENTS = {
    "ds": ds_client,
    "silicon": silicon_client,
    "openrouter": openrouter_client,
    "local":local_client
}
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

MODEL_MAPPING = {
     "eval-1": {
        "provider": "ds",
        "model": "deepseek-chat",
        "price": 
        {
            "in":"2/M token",
            "out":"3/M token"
        }
    },
    "orchestrator-1": {
        "provider": "local",
        "model": "orchestra-8B",
        "price": 
        {
            "in":"0/M token",
            "out":"0/M token"
        }
    },
    "orchestrator-2": {
        "provider": "ds",
        "model": "deepseek-reasoner",
        "price": 
        {
            "in":"2/M token",
            "out":"3/M token"
        }
    },
    "search-1": {
        "provider": "ds",
        "model": "deepseek-chat",
        "price": 
        {
            "in":"2/M token",
            "out":"3/M token"
        }
    },
    "search-2": {
        "provider": "silicon", 
        "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "price": 
        {
            "in":"1/M token",
            "out":"4/M token"
        }
    },
    "search-3": {
        "provider": "silicon",
        "model": "Pro/zai-org/GLM-4.7",
        "price": 
        {
            "in":"14/M token",
            "out":"16/M token"
        }
    },

    "reasoner-1": {
        "provider": "silicon",
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "price": 
        {
            "in":"1.26/M token",
            "out":"1.26/M token"
        }
    },
    "reasoner-2": {
        "provider": "ds",
        "model": "deepseek-chat",
        "price": 
        {
            "in":"2/M token",
            "out":"3/M token"
        }
    },
    "reasoner-3": {
        "provider": "silicon",
        "model": "Pro/zai-org/GLM-4.7",
        "price": 
        {
            "in":"14/M token",
            "out":"16/M token"
        }
    },

    "answer-math-1": {
        "provider": "silicon",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "price": 
        {
            "in":"4.13/M token",
            "out":"4.13/M token"
        }
    },
    "answer-math-2": {
        "provider": "ds",
        "model": "deepseek-chat",
        "price": 
        {
            "in":"2/M token",
            "out":"3/M token"
        }
    },

    "answer-1": {
        "provider": "silicon",
        "model": "Qwen/Qwen3-30B-A3B",
        "price": 
        {
            "in":"0.7/M token",
            "out":"2.8/M token"
        }
    },
    "answer-2": {
        "provider": "ds",
        "model": "deepseek-chat",
        "price": 
        {
            "in":"2/M token",
            "out":"3/M token"
        }
    },
    "answer-3": {
        "provider": "silicon", 
        "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "price": 
        {
            "in":"1/M token",
            "out":"4/M token"
        }
    },
    "answer-4": {
        "provider": "silicon",
        "model": "Pro/zai-org/GLM-4.7",
        "price": 
        {
            "in":"14/M token",
            "out":"16/M token"
        }
    }

}


def get_llm_response(model_alias, messages, tools=None, **kwargs):
    """
    统一的 LLM 调用接口
    """
    config = MODEL_MAPPING.get(model_alias)
    provider = config["provider"]
    real_model_name = config["model"]
    

       
    client = CLIENTS.get(provider)
    if not client:
        return f"Error: Client provider '{provider}' not found."

    print(f"--- Calling Agent: {model_alias} | Provider: {provider} | Model: {real_model_name} ---")
    try:
        
        response = client.chat.completions.create(
                model=real_model_name,
                messages=messages,
                tools=tools,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_length", 8192),
                stream=False,
                timeout=300  
        )
            
        return response
        
    except Exception as e:
        print(f"API Call Error ({real_model_name}): {e}")
        # 返回一个简单的错误标记，或者抛出异常让上层处理
        return str(e)
    