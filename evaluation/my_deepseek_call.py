# my_deepseek_call.py
"""
DeepSeek API调用模块
完全兼容ToolOrchestra的get_llm_response接口
"""

from openai import OpenAI
import os
import time
import json

def get_llm_response(model, messages, temperature=1.0, return_raw_response=False,
                     tools=None, max_length=1024, model_type=None,
                     model_config=None, **kwargs):
    """
    统一的LLM调用接口，全部使用DeepSeek API

    参数:
        model: 模型名称（会被忽略，统一使用DeepSeek）
        messages: 消息列表或字符串
        temperature: 温度参数
        return_raw_response: 是否返回原始响应对象
        tools: OpenAI格式的工具定义
        max_length: 最大生成长度
        其他参数: 兼容性参数，会被忽略
    """

    # 初始化DeepSeek客户端
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # 转换消息格式
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    # DeepSeek模型选择
    # deepseek-chat: 最新的V3模型，性能最好
    deepseek_model = "deepseek-chat"

    # 调用API（带重试）
    answer = ''
    retry_count = 0
    max_retries = 5

    while answer == '' and retry_count < max_retries:
        try:
            # 构建API调用参数
            api_params = {
                "model": deepseek_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": min(max_length, 8000)  # DeepSeek限制
            }

            # 如果有tools参数，添加到请求中
            if tools:
                api_params["tools"] = tools

            # 调用API
            chat_completion = client.chat.completions.create(**api_params)

            # 返回结果
            if return_raw_response:
                answer = chat_completion
            else:
                answer = chat_completion.choices[0].message.content

        except Exception as error:
            retry_count += 1
            print(f'DeepSeek API错误 (重试 {retry_count}/{max_retries}): {error}')

            if retry_count < max_retries:
                # 指数退避
                wait_time = 2 ** retry_count
                print(f'等待 {wait_time} 秒后重试...')
                time.sleep(wait_time)
            else:
                # 最后一次重试失败，返回错误字符串
                print(f'DeepSeek API调用失败，已重试{max_retries}次')
                return "API_ERROR"

    return answer


# 兼容性函数（其他模块可能用到）
def get_openai_client(model):
    """兼容性函数，返回DeepSeek客户端"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
