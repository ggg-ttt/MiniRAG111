#!/usr/bin/env python3
"""
vLLM Server 模式部署脚本
启动 Qwen3-4B-Instruct-2507 模型的 OpenAI 兼容 API 服务器
"""

import os
import sys
import argparse
import subprocess
import signal
import time

# 在导入任何可能初始化 CUDA 的库之前，设置 vLLM 多进程启动方法
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动 vLLM OpenAI API 服务器")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器监听地址，默认 0.0.0.0（所有接口）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器监听端口，默认 8000"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="使用的 GPU 设备，默认 0。可以是单个设备如 '0' 或多个设备如 '0,1'"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="最大模型长度（上下文窗口），默认 8192"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小（多 GPU 时使用），默认 1"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任远程代码（某些模型需要）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型数据类型，默认 auto（自动选择）"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥（可选），用于保护 API 访问"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="vLLM 日志级别，默认 INFO"
    )
    return parser.parse_args()


def start_vllm_server(args):
    """启动 vLLM OpenAI API 服务器"""
    # 设置 CUDA 可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 设置 vLLM 日志级别
    os.environ["VLLM_LOGGING_LEVEL"] = args.log_level
    
    # 构建 vLLM 命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
    ]
    
    # 添加可选参数
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    
    print("=" * 80)
    print("启动 vLLM OpenAI API 服务器")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"GPU: {args.gpu}")
    print(f"最大模型长度: {args.max_model_len}")
    print(f"张量并行大小: {args.tensor_parallel_size}")
    print(f"数据类型: {args.dtype}")
    print(f"日志级别: {args.log_level}")
    print("=" * 80)
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 80)
    print("\n服务器启动中，请稍候...")
    print("启动完成后，可以通过以下方式访问 API：")
    print(f"  - OpenAI 兼容 API: http://{args.host}:{args.port}/v1")
    print(f"  - 文档: http://{args.host}:{args.port}/docs")
    print("\n按 Ctrl+C 停止服务器\n")
    
    # 启动服务器进程
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出日志
        def signal_handler(sig, frame):
            print("\n\n正在停止服务器...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("强制终止服务器...")
                process.kill()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 输出服务器日志
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在停止服务器...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        sys.exit(1)


def main():
    """主函数"""
    args = parse_args()
    start_vllm_server(args)


if __name__ == "__main__":
    main()

