"""
vLLM integration for MiniRAG.

This version follows the same deployment pattern used in `PathRAG/llm.py`,
which loads models directly with the `vllm` runtime instead of exposing an
OpenAI-compatible server.
"""

__version__ = "1.1.0"
__author__ = "lightrag Team"
__status__ = "Production"

import copy
import os
import sys
from functools import lru_cache
from typing import Union

import pipmaster as pm  # 动态安装依赖，避免环境缺包导致导入失败

if not pm.is_installed("torch"):
    pm.install("torch")
if not pm.is_installed("vllm"):
    pm.install("vllm")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import numpy as np
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from vllm import LLM

from minirag.utils import (
    locate_json_string_body_from_string,
    logger,
)

device = "cuda" if torch.cuda.is_available() else "cpu"  # 尽量使用 GPU，加速推理


MS_MODEL_ROOT = "/data/gty/.cache/modelscope/hub/models"  # ModelScope 本地缓存根目录


@lru_cache(maxsize=1)
def initialize_vllm_model(model_name: str):
    """
    Load a vLLM chat model once and cache it for reuse.
    Mirrors the behavior implemented in PathRAG's llm.py.
    """
    ms_local_path = os.path.join(MS_MODEL_ROOT, model_name)
    resolved_model = ms_local_path if os.path.isdir(ms_local_path) else model_name
    logger.info("Initializing vLLM model: %s", resolved_model)
    # vLLM 会在首次调用时常驻显存；使用缓存避免重复加载
    return LLM(model=resolved_model, device=device, max_model_len=8192)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RuntimeError,)),
)
async def vllm_model_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs,
) -> str:
    """
    Run inference with a cached vLLM model.
    The logic is aligned with PathRAG's `vllm_model_if_cache`.
    """
    history_messages = history_messages or []  # 兼容 None
    vllm_model = initialize_vllm_model(model)  # 复用缓存实例

    messages = []  # 构造 Chat 格式输入
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)  # vLLM 原生接口无需该字段，提前剔除

    try:
        result = vllm_model.chat(messages)
    except Exception as err:
        logger.warning("vLLM chat failed with template, retrying: %s", err)
        ori_message = copy.deepcopy(messages)
        if messages and messages[0]["role"] == "system":
            messages[1]["content"] = (
                "<system>" + messages[0]["content"] + "</system>\n" + messages[1]["content"]
            )
            messages = messages[1:]
        result = vllm_model.chat(messages)  # 退化为无 system 的模板
        if not result:
            prompt_text = ""
            len_message = len(ori_message)
            for msgid in range(len_message):
                prompt_text += (
                    f"<{ori_message[msgid]['role']}>"
                    f"{ori_message[msgid]['content']}"
                    f"</{ori_message[msgid]['role']}>\n"
                )
            result = vllm_model.generate(prompt_text)  # 最终兜底：拼接 prompts 走 generate

    if not result:
        raise RuntimeError("vLLM returned empty result")
    return result[0].outputs[0].text


async def vllm_model_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """
    MiniRAG entry-point that mirrors PathRAG's `vllm_model_complete`.
    """
    keyword_extraction = kwargs.pop("keyword_extraction", None)  # Flag 表示是否抽取关键词
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]  # MiniRAG 通过 hashing_kv 传入配置
    result = await vllm_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    return result


async def vllm_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    """
    Simple embedding helper used by PathRAG.
    Keeps the same signature so it can be wired via EmbeddingFunc.
    """
    device = next(embed_model.parameters()).device  # 与建模设备保持一致
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 直接做 mean pooling 简化实现
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    return embeddings.detach().cpu().numpy()