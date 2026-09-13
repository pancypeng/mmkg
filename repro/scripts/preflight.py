#!/usr/bin/env python3
"""Read-only checks required before a CMEL GPU experiment."""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402


MODEL_PATHS = {
    "qwen38-27b": "/root/models/Qwen3.8-27B",
    "qwen38-flash-next-fp8": "/root/models/Qwen3.8-Flash-Next-FP8",
    "stella": "/root/models/stella_en_1.5B_v5",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-api", action="store_true",
        help="只检查本地模型、数据与 GPU；模型服务尚未启动时使用",
    )
    args = parser.parse_args()
    failed = False
    print("[models]")
    for name, path in MODEL_PATHS.items():
        ok = os.path.exists(path) and os.path.exists(os.path.join(path, "config.json"))
        failed |= not ok
        print(f"  {name}: {'OK' if ok else 'MISSING'}  {path}")

    selection = data.select_docs(10 ** 9)
    task2 = images = missing = 0
    for category, docs in selection.items():
        for document in docs:
            task2 += data.task2_gt_count(category, document)
            absent, count = data.check_images(category, document)
            missing += absent
            images += count
    print("[dataset]")
    print("  documents:", {k: len(v) for k, v in selection.items()})
    print(f"  images={images} task2_labels={task2} missing_images={missing}")
    if task2 != 1114 or missing:
        failed = True

    print("[gpus]")
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader",
        ], text=True)
        print(output.rstrip())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  ERROR: {exc}")
        failed = True

    print("[api]")
    if args.skip_api:
        print("  SKIPPED")
    else:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
                models = json.load(response)
            print("  served:", [item["id"] for item in models.get("data", [])])
        except Exception as exc:  # noqa: BLE001 - report any local endpoint failure
            print(f"  NOT READY: {exc}")
            failed = True

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
