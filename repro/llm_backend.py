"""LLM 后端：磁盘缓存 + 重试 + 离线 stub。

原仓库的 pull_llm.py 每次调用都新建 OpenAI client、无缓存、无重试，
一次网络抖动就会让整篇文档的中间结果丢失。这里替换掉这两个入口函数，
并把替换同步到已经 `from pull_llm import ...` 过的模块上。
"""
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time

import config

_lock = threading.Lock()
_mode = "api"
_stats = {
    "hit": 0, "miss": 0, "stub": 0, "error": 0,
    "text_calls": 0, "mm_calls": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    "api_latency_s": 0.0,
}
_client = None
_mm_client = None


# ------------------------------------------------------------------ 缓存
def _cache_path(kind, payload):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    d = os.path.join(config.CACHE_DIR, kind, key[:2])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".json")


def _cache_get(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["response"]
        except Exception:
            return None
    return None


def _cache_put(path, payload, response, raw_response=None, metadata=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        record = {"request": payload, "response": response}
        if raw_response is not None:
            record["raw_response"] = raw_response
        if metadata:
            record["metadata"] = metadata
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp, path)


def _clean_response(response):
    """移除推理块和 Markdown 围栏，但把原始响应保留在缓存记录中。"""
    text = response or ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _request_payload(kind, model, endpoint, system_content, prompt, image_hash=None):
    namespace = (
        config.MM_CACHE_NAMESPACE if kind == "mm" else config.TEXT_CACHE_NAMESPACE
    )
    payload = {
        "schema": 2,
        "namespace": namespace,
        "mode": _mode,
        "kind": kind,
        "endpoint": endpoint.rstrip("/"),
        "model": model,
        "generation": config.generation_config(),
        "system": system_content,
        "prompt": prompt,
    }
    if image_hash is not None:
        payload["image"] = image_hash
    return payload


def _completion_kwargs(model, messages):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "top_p": config.LLM_TOP_P,
        "presence_penalty": config.LLM_PRESENCE_PENALTY,
        "max_tokens": config.LLM_MAX_TOKENS,
        "seed": config.LLM_SEED,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": config.LLM_ENABLE_THINKING,
                "preserve_thinking": False,
            }
        },
    }
    if config.LLM_TOP_K >= 0:
        kwargs["extra_body"]["top_k"] = config.LLM_TOP_K
    if config.LLM_REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
    return kwargs


def _record_completion(kind, completion, latency):
    usage = getattr(completion, "usage", None)
    meta = {
        "latency_s": latency,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "finish_reason": getattr(completion.choices[0], "finish_reason", None),
    }
    with _lock:
        _stats[f"{kind}_calls"] += 1
        _stats["prompt_tokens"] += meta["prompt_tokens"]
        _stats["completion_tokens"] += meta["completion_tokens"]
        _stats["total_tokens"] += meta["total_tokens"]
        _stats["api_latency_s"] += latency
    return meta


# ------------------------------------------------------------------ stub
# 离线模式：返回结构合法但内容无意义的响应，用来验证整条流水线不崩、
# 缓存/聚类/评测都能跑通。准确率必然接近 0，不能当成复现结果。
_STUB_IMAGE_ENTITY = json.dumps({
    "entity_name": "STUB IMAGE ENTITY",
    "entity_type": "EVENT",
    "description": "Offline stub description.",
    "reason": "offline stub",
})


def _stub_response(system_content, prompt, is_mm):
    if is_mm:
        return _STUB_IMAGE_ENTITY
    s = (system_content or "")
    if s.startswith("You are an expert system designed to identify matching entities"):
        # judge_image_entity_alignment -> 期望返回一个实体名字符串
        return "STUB MATCHED ENTITY"
    # 其余几条（text_entity_judgement / llm_*_alignment / clustering 融合判定）
    # 都期望一个 JSON 列表
    return "[]"


# ------------------------------------------------------------------ API
def _get_client(mm):
    global _client, _mm_client
    from openai import OpenAI
    with _lock:
        if mm:
            if _mm_client is None:
                _mm_client = OpenAI(base_url=config.MM_API_BASE, api_key=config.MM_API_KEY,
                                    timeout=config.LLM_TIMEOUT)
            return _mm_client
        if _client is None:
            _client = OpenAI(base_url=config.API_BASE, api_key=config.API_KEY,
                             timeout=config.LLM_TIMEOUT)
        return _client


def _call_with_retry(fn):
    last = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            return fn()
        except Exception as e:          # noqa: BLE001 - 网络/限流都在这里兜住
            last = e
            with _lock:
                _stats["error"] += 1
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} attempts: {last}")


def get_llm_response(cur_prompt, system_content):
    payload = _request_payload(
        "text", config.MODEL, config.API_BASE, system_content, cur_prompt
    )
    path = _cache_path("text", payload)
    cached = _cache_get(path)
    if cached is not None:
        with _lock:
            _stats["hit"] += 1
        return cached

    if _mode == "stub":
        resp = _stub_response(system_content, cur_prompt, is_mm=False)
        with _lock:
            _stats["stub"] += 1
        _cache_put(path, payload, resp)
        return resp

    def _do():
        c = _get_client(mm=False)
        return c.chat.completions.create(**_completion_kwargs(
            config.MODEL,
            [{"role": "system", "content": system_content},
             {"role": "user", "content": cur_prompt}],
        ))

    t0 = time.perf_counter()
    completion = _call_with_retry(_do)
    latency = time.perf_counter() - t0
    raw = completion.choices[0].message.content or ""
    resp = _clean_response(raw)
    meta = _record_completion("text", completion, latency)
    with _lock:
        _stats["miss"] += 1
    _cache_put(path, payload, resp, raw_response=raw, metadata=meta)
    return resp


def get_mmllm_response(cur_prompt, system_content, img_base):
    # 缓存 key 用图片内容哈希，避免把整张 base64 写进 request 记录
    img_hash = hashlib.sha256(img_base.encode("utf-8")).hexdigest()
    payload = _request_payload(
        "mm", config.MM_MODEL, config.MM_API_BASE, system_content, cur_prompt,
        image_hash=img_hash,
    )
    path = _cache_path("mm", payload)
    cached = _cache_get(path)
    if cached is not None:
        with _lock:
            _stats["hit"] += 1
        return cached

    if _mode == "stub":
        resp = _stub_response(system_content, cur_prompt, is_mm=True)
        with _lock:
            _stats["stub"] += 1
        _cache_put(path, payload, resp)
        return resp

    def _do():
        c = _get_client(mm=True)
        return c.chat.completions.create(**_completion_kwargs(
            config.MM_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}},
                    {"type": "text", "text": cur_prompt},
                ]},
            ],
        ))

    t0 = time.perf_counter()
    completion = _call_with_retry(_do)
    latency = time.perf_counter() - t0
    raw = completion.choices[0].message.content or ""
    resp = _clean_response(raw)
    meta = _record_completion("mm", completion, latency)
    with _lock:
        _stats["miss"] += 1
    _cache_put(path, payload, resp, raw_response=raw, metadata=meta)
    return resp


# ------------------------------------------------------------------ 注入
def install(mode="api"):
    """替换 pull_llm 的两个入口，并同步到已 import 过它的模块。

    methods.py / clustering_function.py 用的是 `from pull_llm import f`，
    绑定的是函数对象本身，所以只改 pull_llm 模块属性是不够的。
    """
    global _mode
    _mode = mode
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    if config.CMEL_RESEARCH not in sys.path:
        sys.path.insert(0, config.CMEL_RESEARCH)

    import pull_llm
    pull_llm.API_KEY, pull_llm.URL, pull_llm.MODEL = config.API_KEY, config.API_BASE, config.MODEL
    pull_llm.MM_API_KEY, pull_llm.MM_URL, pull_llm.MM_MODEL = config.MM_API_KEY, config.MM_API_BASE, config.MM_MODEL
    pull_llm.get_llm_response = get_llm_response
    pull_llm.get_mmllm_response = get_mmllm_response

    import methods
    import clustering_function
    methods.get_llm_response = get_llm_response
    methods.get_mmllm_response = get_mmllm_response
    clustering_function.get_llm_response = get_llm_response
    return methods, clustering_function


def stats():
    with _lock:
        out = dict(_stats)
    out["api_latency_s"] = round(out["api_latency_s"], 3)
    return out
