"""Build an auditable union of existing CMEL predictions without model calls.

The published Task 2 metric ignores false positives, so this is primarily a protocol audit.
Always inspect the generated task2_error_analysis.md precision/F1 before interpreting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_task2_errors
import config
import evaluate
from run_cmel import write_json_atomic


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prediction_key(item):
    return (tuple(sorted(set(item.get("source_image_entities") or []))),
            tuple(sorted(set(item.get("source_text_entities") or []))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Output run name")
    parser.add_argument("--source", action="append", required=True, help="Source run, repeat >=2")
    args = parser.parse_args()
    if len(args.source) < 2:
        parser.error("at least two --source runs are required")

    source_dirs = [Path(config.RUNS_DIR) / name for name in args.source]
    selections = [read_json(path / "selection.json") for path in source_dirs]
    if any(selection != selections[0] for selection in selections[1:]):
        raise SystemExit("source runs use different document selections")

    out_dir = Path(config.RUNS_DIR) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": "ensemble_union_protocol_audit",
        "source_runs": args.source,
        "selection_source": args.source[0],
        "warning": "Published Task 2 accuracy ignores false positives; inspect precision/F1.",
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists() and read_json(config_path) != run_config:
        raise SystemExit(f"run {args.run!r} exists with a different configuration")
    write_json_atomic(str(config_path), run_config)
    write_json_atomic(str(out_dir / "selection.json"), selections[0])

    images = 0
    for category, documents in selections[0].items():
        for document in documents:
            source_results = [read_json(path / category / document / "result.json")
                              for path in source_dirs]
            image_ids = list(source_results[0])
            if any(set(result) != set(image_ids) for result in source_results[1:]):
                raise SystemExit(f"image keys differ for {category}/{document}")
            combined = {}
            for image_id in image_ids:
                primary = source_results[0][image_id]
                merged = []
                seen = set()
                for result in source_results:
                    record = result.get(image_id) or {}
                    for item in record.get("merged_entities") or []:
                        if not isinstance(item, dict):
                            continue
                        key = prediction_key(item)
                        if not key[0] or not key[1] or key in seen:
                            continue
                        seen.add(key)
                        merged.append(item)
                combined[image_id] = {
                    "matched_entity": (primary or {}).get("matched_entity", "no match"),
                    "merged_entities": merged,
                }
                images += 1
            doc_dir = out_dir / category / document
            doc_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(str(doc_dir / "result.json"), combined)

    results = evaluate.evaluate_run(str(out_dir))
    write_json_atomic(str(out_dir / "metrics.json"), results)
    (out_dir / "report.md").write_text(
        evaluate.format_report(str(out_dir), results), encoding="utf-8")
    write_json_atomic(str(out_dir / "progress.json"), {
        "status": "complete", "images_completed": images, "images_expected": images,
        "source_runs": args.source, "model_calls": 0,
    })
    analysis = analyze_task2_errors.analyze_run(out_dir)
    payload = {"primary": analysis}
    (out_dir / "task2_error_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "task2_error_analysis.md").write_text(
        analyze_task2_errors.markdown(payload), encoding="utf-8")
    print(evaluate.format_report(str(out_dir), results))
    print("\n" + analyze_task2_errors.markdown(payload))


if __name__ == "__main__":
    main()
