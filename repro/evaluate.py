"""评测：与 fusion_research.py 的 calculate_task{1,2,3}_accuracy 逻辑一致，
只是把「标注在数据集里、结果在 run 目录里」的路径拆开。

与原实现的唯一行为差异：原实现在 result.json 缺某张图片时会 KeyError 崩溃，
这里记为答错并计入 missing 统计（小样本/中断续跑时更有用）。
"""
import argparse
import json
import os
from difflib import SequenceMatcher

import config
import data


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def task1_accuracy(gt, result):
    total = correct = missing = 0
    for image_key in gt:
        total += 1
        if image_key not in result or not isinstance(result[image_key], dict):
            missing += 1
            continue
        aligned_entity = (gt[image_key].get("matched_chunk_entity_name") or "").lower()
        result_entity = (result[image_key].get("matched_entity") or "").lower()
        if aligned_entity == result_entity:
            correct += 1
        elif aligned_entity and result_entity and (
                aligned_entity in result_entity or result_entity in aligned_entity):
            correct += 1
        elif SequenceMatcher(None, aligned_entity, result_entity).ratio() > 0.5:
            correct += 1
    return (correct / total if total else 0.0), total, missing


def task2_accuracy(gt, result):
    total = correct = 0
    for image_key, aligned_entities in gt.items():
        res = result.get(image_key)
        result_entities = (res or {}).get("merged_entities") or [] if isinstance(res, dict) else []
        for aligned_entity in aligned_entities:
            a_img = set(aligned_entity["source_image_entities"])
            a_txt = set(aligned_entity["source_text_entities"])
            for r in result_entities:
                if not isinstance(r, dict):
                    continue
                if (set(r.get("source_image_entities", [])) == a_img
                        and set(r.get("source_text_entities", [])) == a_txt):
                    correct += 1
                    break
            total += 1
    return (correct / total if total else 0.0), total, 0


def task3_accuracy(gt, result):
    total = correct = missing = 0
    for image_key in gt:
        total += 1
        res = result.get(image_key)
        if not isinstance(res, dict):
            missing += 1
            continue
        aligned_entity = (gt[image_key].get("entity_name") or "").lower()
        result_entity = (res.get("entity_name") or "").lower()
        if aligned_entity == result_entity:
            correct += 1
        elif aligned_entity and result_entity and (
                aligned_entity in result_entity or result_entity in aligned_entity):
            correct += 1
        elif SequenceMatcher(None, aligned_entity, result_entity).ratio() > 0.7:
            correct += 1
    return (correct / total if total else 0.0), total, missing


def _blank():
    return {c: {"acc": [], "correct": 0.0, "total": 0, "missing": 0} for c in config.CATEGORIES}


def _agg(stats):
    out, all_acc, correct, total, missing = {}, [], 0.0, 0, 0
    for cat, s in stats.items():
        macro = sum(s["acc"]) / len(s["acc"]) if s["acc"] else 0.0
        micro = s["correct"] / s["total"] if s["total"] else 0.0
        out[cat] = {"micro": micro, "macro": macro, "total": s["total"],
                    "docs": len(s["acc"]), "missing": s["missing"]}
        all_acc += s["acc"]
        correct += s["correct"]
        total += s["total"]
        missing += s["missing"]
    return {"overall": {"micro": correct / total if total else 0.0,
                        "macro": sum(all_acc) / len(all_acc) if all_acc else 0.0,
                        "total": total, "docs": len(all_acc), "missing": missing},
            "per_category": out}


def evaluate_run(run_dir, tasks=None):
    if tasks is None:
        cfg_path = os.path.join(run_dir, "run_config.json")
        method = (_read(cfg_path).get("method") if os.path.exists(cfg_path) else None)
        if method == "task3":
            tasks = ("task3",)
        elif method in {"embedding", "llm", "clustering",
                        "ensemble_union_protocol_audit",
                        "ensemble_candidate_adjudication_v1"}:
            tasks = ("task1", "task2")
        else:
            # 无 run_config 的 evaluator oracle/null 自检仍覆盖全部三个任务。
            tasks = ("task1", "task2", "task3")
    stats = {t: _blank() for t in tasks}
    for cat in config.CATEGORIES:
        cat_dir = os.path.join(run_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for doc in sorted(os.listdir(cat_dir)):
            result_path = os.path.join(cat_dir, doc, "result.json")
            if not os.path.exists(result_path):
                continue
            result = _read(result_path)
            gt1, gt2 = data.load_gt(cat, doc)
            for task, fn, gt in (("task1", task1_accuracy, gt1),
                                 ("task2", task2_accuracy, gt2),
                                 ("task3", task3_accuracy, gt1)):
                if task not in stats:
                    continue
                acc, total, missing = fn(gt, result)
                if total == 0:
                    continue
                s = stats[task][cat]
                s["acc"].append(acc)
                s["correct"] += acc * total
                s["total"] += total
                s["missing"] += missing
    return {t: _agg(s) for t, s in stats.items()}


# 论文 Table 1（micro/macro %）：News / Aca. / Nov. / Overall
PAPER_TABLE1 = {
    "Emb":  [(10.8, 8.4), (33.1, 34.5), (9.0, 7.5), (20.0, 16.8)],
    "LLM":  [(33.3, 24.1), (36.8, 36.1), (17.4, 20.8), (27.1, 27.0)],
    "DB":   [(53.8, 45.9), (60.8, 58.3), (29.9, 34.2), (45.2, 46.1)],
    "KM":   [(50.5, 40.6), (60.7, 57.7), (29.6, 30.5), (45.2, 43.0)],
    "PR":   [(51.6, 44.4), (59.7, 56.8), (29.1, 35.2), (44.1, 45.5)],
    "Lei":  [(54.8, 44.7), (60.5, 55.5), (29.4, 30.6), (44.8, 43.6)],
    "Spec": [(65.5, 56.9), (73.3, 69.9), (31.2, 39.4), (51.8, 59.2)],
}


def format_report(run_dir, results):
    lines = [f"# CMEL 复现结果  ({os.path.basename(run_dir)})", ""]
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        lines += ["```json", json.dumps(_read(cfg_path), indent=2, ensure_ascii=False), "```", ""]
    for task, r in results.items():
        o = r["overall"]
        if o["total"] == 0:
            continue
        lines += [f"## {task}", "",
                  "| 域 | micro % | macro % | 样本数 | 文档数 | 缺失 |",
                  "|---|---|---|---|---|---|"]
        for cat in config.CATEGORIES:
            c = r["per_category"][cat]
            lines.append(f"| {cat} | {c['micro']*100:.1f} | {c['macro']*100:.1f} | "
                         f"{c['total']} | {c['docs']} | {c['missing']} |")
        lines.append(f"| **overall** | **{o['micro']*100:.1f}** | **{o['macro']*100:.1f}** | "
                     f"{o['total']} | {o['docs']} | {o['missing']} |")
        lines.append("")
        if task == "task2":
            paper_micro, paper_macro = PAPER_TABLE1["Spec"][-1]
            delta_micro = o["micro"] * 100 - paper_micro
            delta_macro = o["macro"] * 100 - paper_macro
            coverage = "完整集" if o["total"] == 1114 else "子集（不可作为正式超越结论）"
            lines += [
                f"论文 Spe-L 最佳值：**{paper_micro:.1f}/{paper_macro:.1f}**（micro/macro）。",
                f"本次差值：**{delta_micro:+.1f}/{delta_macro:+.1f}** 个百分点；评测覆盖：{coverage}。",
                "",
            ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(config.RUNS_DIR, args.run)
    results = evaluate_run(run_dir)
    report = format_report(run_dir, results)
    print(report)
    out = args.out or os.path.join(run_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
