import re
import json
from openai import OpenAI

API_KEY = ""
MODEL = ""
URL = ""
MM_API_KEY = ""
MM_MODEL = ""
MM_URL = ""

# 正则化处理函数
def _json_values(output):
    """从带代码围栏/解释文字的输出中提取完整 JSON 值，支持嵌套结构。"""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", output or "", flags=re.I)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.I).strip()
    decoder = json.JSONDecoder()
    for candidate in (cleaned, cleaned.replace('\\"', '"')):
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            yield value


def _repair_json_value(output):
    """只在标准 JSON decoder 失败后修复常见引号/转义错误。"""
    try:
        from json_repair import repair_json
    except ImportError:
        return None
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", output or "", flags=re.I)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.I).strip()
    try:
        return repair_json(cleaned, return_objects=True)
    except Exception:
        return None


def normalize_to_json(output):
    for value in _json_values(output):
        if isinstance(value, dict):
            return value
    repaired = _repair_json_value(output)
    if isinstance(repaired, dict):
        return repaired
    print("未找到有效的 JSON 对象")
    return None


def normalize_to_json_list(output):
    for value in _json_values(output):
        if isinstance(value, list):
            return value

    repaired = _repair_json_value(output)
    if isinstance(repaired, list):
        return repaired

    # 截断列表的降级处理：只回收其中能够独立解析的对象。
    objects = [value for value in _json_values(output) if isinstance(value, dict)]
    if objects:
        print("JSON 列表不完整，已回收可解析对象")
        return objects
    print("未找到有效的 JSON 列表")
    return []

# 用LLM进行回答
def get_llm_response(cur_prompt, system_content):
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": cur_prompt},
        ],
    )

    response = completion.choices[0].message.content
    return response

# 调用多模态LLM
def get_mmllm_response(cur_prompt, system_content, img_base):
    client = OpenAI(
        base_url=MM_URL, api_key=MM_API_KEY
    )

    completion = client.chat.completions.create(
        model=MM_MODEL,
        messages=[
            {"role": "system", "content": [
                    {
                        "type": "text",
                        "text": system_content
                    }
                    ]},
            {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base}"},
                    },
                    {
                        "type": "text",
                        "text": cur_prompt
                    }
                    ]},
        ],
    )

    response = completion.choices[0].message.content
    return response
