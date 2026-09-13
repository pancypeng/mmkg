"""Analyze exact-match CMEL Task 2 errors and optional cross-run complementarity."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import config
import data


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def entity_sets(item):
    return (frozenset(item.get("source_image_entities") or []),
            frozenset(item.get("source_text_entities") or []))


def overlap(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def classify(gold, predictions):
    gold_image, gold_text = entity_sets(gold)
    pairs = [(pred, *entity_sets(pred)) for pred in predictions if isinstance(pred, dict)]
    if any(image == gold_image and text == gold_text for _, image, text in pairs):
        return "exact", 1.0
    image_exact = any(image == gold_image for _, image, _ in pairs)
    text_exact = any(text == gold_text for _, _, text in pairs)
    if image_exact and text_exact:
        bucket = "both_sides_exact_but_not_paired"
    elif image_exact:
        bucket = "image_exact_text_wrong"
    elif text_exact:
        bucket = "text_exact_image_wrong"
    else:
        best = max((0.5 * overlap(gold_image, image) + 0.5 * overlap(gold_text, text)
                    for _, image, text in pairs), default=0.0)
        bucket = "partial_overlap" if best > 0 else "no_overlap"
        return bucket, best
    best = max((0.5 * overlap(gold_image, image) + 0.5 * overlap(gold_text, text)
                for _, image, text in pairs), default=0.0)
    return bucket, best


def analyze_run(run_dir: Path):
    rows = []
    doc_scores = []
    prediction_cardinality = Counter()
    gold_cardinality = Counter()
    link_counts = {category: Counter() for category in config.CATEGORIES}
    for category in config.CATEGORIES:
        category_dir = run_dir / category
        if not category_dir.is_dir():
            continue
        for doc_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            result_path = doc_dir / "result.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            _, gold_data = data.load_gt(category, doc_dir.name)
            doc_correct = 0
            doc_total = 0
            for image_id, gold_items in gold_data.items():
                res = result.get(image_id) if isinstance(result, dict) else None
                predictions = ((res or {}).get("merged_entities") or []
                               if isinstance(res, dict) else [])
                gold_pairs = {entity_sets(item) for item in gold_items}
                predicted_pairs = {entity_sets(item) for item in predictions
                                   if isinstance(item, dict)
                                   and item.get("source_image_entities")
                                   and item.get("source_text_entities")}
                link_counts[category]["gold"] += len(gold_pairs)
                link_counts[category]["predicted"] += len(predicted_pairs)
                link_counts[category]["true_positive"] += len(gold_pairs & predicted_pairs)
                prediction_cardinality[len(predictions)] += 1
                for gold in gold_items:
                    bucket, best_overlap = classify(gold, predictions)
                    gold_image, gold_text = entity_sets(gold)
                    gold_cardinality[(len(gold_image), len(gold_text))] += 1
                    doc_total += 1
                    doc_correct += bucket == "exact"
                    rows.append({
                        "category": category,
                        "document": doc_dir.name,
                        "image_id": image_id,
                        "gold": gold,
                        "predictions": predictions,
                        "bucket": bucket,
                        "best_overlap": round(best_overlap, 6),
                    })
            if doc_total:
                doc_scores.append({"category": category, "document": doc_dir.name,
                                   "correct": doc_correct, "total": doc_total,
                                   "accuracy": doc_correct / doc_total})
    buckets = Counter(row["bucket"] for row in rows)
    per_category = {}
    for category in config.CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]
        counts = Counter(row["bucket"] for row in category_rows)
        per_category[category] = {
            "total": len(category_rows),
            "exact": counts["exact"],
            "accuracy": counts["exact"] / len(category_rows) if category_rows else 0.0,
            "buckets": dict(counts),
            "link_metrics": link_metrics(link_counts[category]),
        }
    examples = {}
    for bucket in buckets:
        if bucket == "exact":
            continue
        candidates = sorted((row for row in rows if row["bucket"] == bucket),
                            key=lambda row: (-row["best_overlap"], row["category"],
                                             row["document"], row["image_id"]))
        examples[bucket] = candidates[:3]
    overall_link_counts = Counter()
    for counts in link_counts.values():
        overall_link_counts.update(counts)
    return {
        "run": run_dir.name,
        "total": len(rows),
        "exact": buckets["exact"],
        "accuracy": buckets["exact"] / len(rows) if rows else 0.0,
        "buckets": dict(buckets),
        "per_category": per_category,
        "link_metrics": link_metrics(overall_link_counts),
        "gold_set_cardinality": {f"image_{i}_text_{t}": n
                                 for (i, t), n in sorted(gold_cardinality.items())},
        "predictions_per_image": {str(k): v for k, v in sorted(prediction_cardinality.items())},
        "doc_scores": sorted(doc_scores, key=lambda row: (row["accuracy"], row["category"],
                                                            row["document"])),
        "examples": examples,
        "rows": rows,
    }


def link_metrics(counts):
    true_positive = counts["true_positive"]
    predicted = counts["predicted"]
    gold = counts["gold"]
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": true_positive, "predicted": predicted, "gold": gold,
            "false_positive": predicted - true_positive, "false_negative": gold - true_positive,
            "precision": precision, "recall": recall, "f1": f1}


def compare_runs(primary, others):
    all_runs = [primary, *others]
    maps = []
    for run in all_runs:
        row_map = {}
        for row in run["rows"]:
            key = (row["category"], row["document"], row["image_id"],
                   json.dumps(row["gold"], ensure_ascii=False, sort_keys=True))
            row_map[key] = row["bucket"] == "exact"
        maps.append(row_map)
    keys = set(maps[0])
    for row_map in maps[1:]:
        keys &= set(row_map)
    patterns = Counter(tuple(row_map[key] for row_map in maps) for key in keys)
    oracle = sum(any(row_map[key] for row_map in maps) for key in keys)
    return {
        "runs": [run["run"] for run in all_runs],
        "shared_total": len(keys),
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / len(keys) if keys else 0.0,
        "correctness_patterns": {"/".join("1" if bit else "0" for bit in pattern): count
                                 for pattern, count in sorted(patterns.items())},
    }


def compact_item(item):
    image, text = entity_sets(item)
    return f"img={sorted(image)}; text={sorted(text)}"


def markdown(payload):
    primary = payload["primary"]
    lines = [f"# Task 2 误差分析：{primary['run']}", "",
             "评测仍使用论文的严格集合完全匹配；该报告不修改预测或金标准。", "",
             "## 总览", "",
             "| 域 | correct/total | accuracy |", "|---|---:|---:|"]
    for category, summary in primary["per_category"].items():
        lines.append(f"| {category} | {summary['exact']}/{summary['total']} | "
                     f"{summary['accuracy']*100:.1f}% |")
    lines += [f"| **overall** | **{primary['exact']}/{primary['total']}** | "
              f"**{primary['accuracy']*100:.1f}%** |", "", "## 错误分解", "",
              "| 类别 | 数量 | 占全部 gold |", "|---|---:|---:|"]
    order = ["exact", "image_exact_text_wrong", "text_exact_image_wrong",
             "both_sides_exact_but_not_paired", "partial_overlap", "no_overlap"]
    for bucket in order:
        count = primary["buckets"].get(bucket, 0)
        lines.append(f"| `{bucket}` | {count} | {count/primary['total']*100:.1f}% |")
    link = primary["link_metrics"]
    lines += ["", "## 假阳性敏感的链接指标", "",
              "论文 accuracy 不惩罚额外预测；下表把每个图片内唯一的图像/文本集合对作为预测链接。",
              "", "| TP | predicted | gold | FP | FN | precision | recall | F1 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|",
              f"| {link['true_positive']} | {link['predicted']} | {link['gold']} | "
              f"{link['false_positive']} | {link['false_negative']} | "
              f"{link['precision']*100:.1f}% | {link['recall']*100:.1f}% | "
              f"{link['f1']*100:.1f}% |"]
    if payload.get("comparison"):
        comparison = payload["comparison"]
        lines += ["", "## 跨 run 严格 oracle（仅诊断）", "",
                  f"Runs：`{'`, `'.join(comparison['runs'])}`。", "",
                  f"共同样本 {comparison['shared_total']}；任一 run 答对即算对的 oracle 为 "
                  f"**{comparison['oracle_correct']}/{comparison['shared_total']} = "
                  f"{comparison['oracle_accuracy']*100:.1f}%**。这不是可报告模型分数。", "",
                  "| correctness pattern | 数量 |", "|---|---:|"]
        for pattern, count in comparison["correctness_patterns"].items():
            lines.append(f"| `{pattern}` | {count} |")
    lines += ["", "## 最差文档", "", "| 域/文档 | correct/total | accuracy |",
              "|---|---:|---:|"]
    for row in primary["doc_scores"][:10]:
        lines.append(f"| {row['category']}/{row['document']} | {row['correct']}/{row['total']} | "
                     f"{row['accuracy']*100:.1f}% |")
    lines += ["", "## 典型错误", ""]
    for bucket, examples in primary["examples"].items():
        lines += [f"### `{bucket}`", ""]
        for row in examples:
            prediction_text = "<empty>" if not row["predictions"] else " | ".join(
                compact_item(item) for item in row["predictions"] if isinstance(item, dict))
            lines += [f"- `{row['category']}/{row['document']}/{row['image_id']}` "
                      f"(best overlap={row['best_overlap']:.3f})", "",
                      f"  - gold: {compact_item(row['gold'])}",
                      f"  - predicted: {prediction_text}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Primary run name under runs/")
    parser.add_argument("--compare", action="append", default=[], help="Additional run name")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    primary = analyze_run(Path(config.RUNS_DIR) / args.run)
    comparisons = [analyze_run(Path(config.RUNS_DIR) / name) for name in args.compare]
    payload = {"primary": primary}
    if comparisons:
        payload["comparison"] = compare_runs(primary, comparisons)
    prefix = Path(args.out_prefix or (Path(config.RUNS_DIR) / args.run / "task2_error_analysis"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.loads(json.dumps(payload))
    with (prefix.with_suffix(".json")).open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)
    with (prefix.with_suffix(".md")).open("w", encoding="utf-8") as handle:
        handle.write(markdown(payload))
    print(markdown(payload))


if __name__ == "__main__":
    main()
