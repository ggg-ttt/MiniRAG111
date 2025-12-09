#!/usr/bin/env python3
"""
vLLM Server 使用示例
演示如何连接到 vLLM OpenAI API 服务器并调用模型
"""

import requests
import json
import argparse

def chat_completion(api_base="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507", 
                    messages=None, api_key=None, **kwargs):
    """
    调用 vLLM OpenAI API 进行对话
    
    Args:
        api_base: API 服务器地址
        model: 模型名称
        messages: 消息列表，格式: [{"role": "user", "content": "..."}]
        api_key: API 密钥（如果设置了）
        **kwargs: 其他参数（temperature, max_tokens 等）
    
    Returns:
        模型响应文本
    """
    url = f"{api_base}/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    data = {
        "model": model,
        "messages": messages or [],
        **kwargs
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=300)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    """示例：使用 vLLM API 服务器"""
    parser = argparse.ArgumentParser(description="vLLM API 服务器使用示例")
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API 服务器地址"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="模型名称"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥（如果设置了）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请介绍一下你自己。",
        help="要发送的提示文本"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("vLLM API 服务器使用示例")
    print("=" * 80)
    print(f"API 地址: {args.api_base}")
    print(f"模型: {args.model}")
    print(f"提示: {args.prompt}")
    print("=" * 80)
    
    try:
        messages = [
            {"role": "user", "content": args.prompt}
        ]
        
        print("\n正在调用 API...")
        response = chat_completion(
            api_base=args.api_base,
            model=args.model,
            messages=messages,
            api_key=args.api_key,
            temperature=0.7,
            max_tokens=512
        )
        
        print("\n模型响应:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到 API 服务器！")
        print("请确保 vLLM 服务器正在运行。")
        print("启动服务器: python reproduce/vllm_server.py")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()

