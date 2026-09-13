#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible Qwen VLM with text and one local image."""
import argparse
import base64
import json
import mimetypes
import os
import time

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    args = parser.parse_args()

    mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
    with open(args.image, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180)
    common = {
        "model": args.model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
        "seed": 0,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            }
        },
    }
    started = time.perf_counter()
    text_response = client.chat.completions.create(
        **common,
        messages=[
            {"role": "system", "content": "Follow the user exactly."},
            {"role": "user", "content": 'Return exactly this JSON object: {"ok": true}'},
        ],
    )
    image_response = client.chat.completions.create(
        **common,
        messages=[
            {"role": "system", "content": "Be concise and factual."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{encoded}",
                }},
                {"type": "text", "text": "Describe the main people and situation in one sentence."},
            ]},
        ],
    )
    record = {
        "model": args.model,
        "base_url": args.base_url,
        "image": os.path.abspath(args.image),
        "text_response": text_response.choices[0].message.content,
        "image_response": image_response.choices[0].message.content,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    output = json.dumps(record, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        tmp = args.output + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")
        os.replace(tmp, args.output)


if __name__ == "__main__":
    main()
