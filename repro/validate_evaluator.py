"""离线校验评测代码本身是否正确。

不调任何 API：直接用标注构造两组假结果，
  - oracle：完全照抄标注 -> 三个 task 都应该是 100%
  - null  ：全部空/错 -> 三个 task 都应该是 0%
如果这两条不成立，说明评测口径写错了，后面跑出来的数字没有意义。
"""
import json
import os
import shutil
import sys

import config
import data
import evaluate


def build_fake_run(run_name, kind, selection):
    run_dir = os.path.join(config.RUNS_DIR, run_name)
    shutil.rmtree(run_dir, ignore_errors=True)
    for cat, docs in selection.items():
        for doc in docs:
            gt1, gt2 = data.load_gt(cat, doc)
            result = {}
            for k, v in gt1.items():
                if kind == "oracle":
                    result[k] = {"matched_entity": v.get("matched_chunk_entity_name", ""),
                                 "entity_name": v.get("entity_name", ""),
                                 "merged_entities": []}
                else:
                    result[k] = {"matched_entity": "zzz_no_such_entity_zzz",
                                 "entity_name": "zzz_no_such_entity_zzz",
                                 "merged_entities": []}
            if kind == "oracle":
                for k, entries in gt2.items():
                    result.setdefault(k, {"matched_entity": "", "entity_name": "",
                                          "merged_entities": []})
                    result[k]["merged_entities"] = [
                        {"source_image_entities": e["source_image_entities"],
                         "source_text_entities": e["source_text_entities"]}
                        for e in entries]
            out = os.path.join(run_dir, cat, doc)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    return run_dir


def main():
    selection = data.select_docs(int(sys.argv[1]) if len(sys.argv) > 1 else 4, seed=0)
    n = sum(len(v) for v in selection.values())
    print(f"抽样文档: {n} 篇 -> " + ", ".join(f"{c}:{len(v)}" for c, v in selection.items()))

    ok = True
    for kind, expect in (("oracle", 1.0), ("null", 0.0)):
        run_dir = build_fake_run(f"_validate_{kind}", kind, selection)
        res = evaluate.evaluate_run(run_dir)
        for task in ("task1", "task2", "task3"):
            o = res[task]["overall"]
            if o["total"] == 0:
                continue
            micro, macro = o["micro"], o["macro"]
            good = abs(micro - expect) < 1e-9 and abs(macro - expect) < 1e-9
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {kind:6s} {task}: "
                  f"micro={micro*100:.1f}% macro={macro*100:.1f}% "
                  f"(样本 {o['total']}, 期望 {expect*100:.0f}%)")
        shutil.rmtree(run_dir, ignore_errors=True)

    print("\n评测口径校验:", "通过" if ok else "未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
