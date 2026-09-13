"""LLM adjudication over the exact candidate links emitted by existing CMEL runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from json_repair import repair_json

import analyze_task2_errors
import config
import data
import evaluate
import llm_backend
from build_union_ensemble import prediction_key, read_json
from run_cmel import write_json_atomic


SYSTEM_PROMPT = """You are a high-precision cross-modal entity-linking adjudicator.
Select valid candidate links without rewriting or inventing links. Return strict JSON only."""


def parse_keep_ids(raw, allowed):
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        value = repair_json(raw, return_objects=True)
    if isinstance(value, dict):
        value = value.get("keep_ids") or []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item in allowed]


def union_candidates(source_results, image_id, source_names):
    candidates = []
    by_key = {}
    for source_name, result in zip(source_names, source_results):
        record = result.get(image_id) or {}
        for item in record.get("merged_entities") or []:
            if not isinstance(item, dict):
                continue
            key = prediction_key(item)
            if not key[0] or not key[1]:
                continue
            if key not in by_key:
                candidate = {
                    "candidate_id": f"c{len(candidates)}",
                    "source_image_entities": list(key[0]),
                    "source_text_entities": list(key[1]),
                    "entity_name": item.get("entity_name", ""),
                    "description": item.get("description", ""),
                    "proposed_by": [],
                    "_record": item,
                }
                candidates.append(candidate)
                by_key[key] = candidate
            by_key[key]["proposed_by"].append(source_name)
    return candidates


def build_prompt(candidates, image_entities, text_entities, nearby_chunks):
    image_map = {item.get("entity_name"): {
        "entity_type": item.get("entity_type", "N/A"),
        "description": item.get("description", ""),
    } for item in image_entities}
    text_map = {item.get("entity_name"): {
        "entity_type": item.get("entity_type", "N/A"),
        "description": item.get("description", ""),
    } for item in text_entities}
    used_image_names = {name for item in candidates for name in item["source_image_entities"]}
    used_text_names = {name for item in candidates for name in item["source_text_entities"]}
    context = "\n\n".join(str(chunk) for chunk in nearby_chunks)[:12000]
    public_candidates = [{key: value for key, value in item.items() if key != "_record"}
                         for item in candidates]
    payload = {
        "candidates": public_candidates,
        "image_entity_evidence": {name: image_map.get(name, {})
                                  for name in sorted(used_image_names)},
        "text_entity_evidence": {name: text_map.get(name, {})
                                 for name in sorted(used_text_names)},
        "nearby_text_context": context,
    }
    return f"""
Two independently generated CMEL predictions have been deduplicated into candidate links. Decide
which candidate records are supported by the entity descriptions and nearby text.

{json.dumps(payload, ensure_ascii=False)}

CMEL conventions:
1. A visual object may link to a non-literal but best contextual anchor (person, location, event,
   method, or concept) when the text does not name the visible object literally.
2. source_text_entities may contain multiple names only when all are aliases/coreferential mentions
   required for the same target. Reject candidates padded with merely related causes, locations,
   authors, methods, components, or co-mentioned objects.
3. Multiple source_image_entities are valid only when they jointly denote the same plural/collective
   textual target or are inseparable parts of one target. Reject accidental grouping of unrelated
   visual objects.
4. Agreement by both generators is useful evidence but is not proof. A candidate from only one
   generator may still be correct. Judge each candidate independently and preserve supported links.
5. You may only select candidate_id values shown above. Do not create, split, combine, or rewrite a
   candidate.

Return exactly {{"keep_ids": ["c0", "c2"]}}. Return an empty list when none are supported.
"""


def make_report(run_dir, results, analysis):
    base = evaluate.format_report(str(run_dir), results)
    return base + "\n\n" + analyze_task2_errors.markdown({"primary": analysis})


def count_checkpointed_images(run_dir):
    total = 0
    for result_path in run_dir.glob("*/*/result.json"):
        try:
            total += len(read_json(result_path))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return total


def sum_stats(previous, current):
    return {key: (round(previous.get(key, 0) + value, 3)
                  if isinstance(value, float) else previous.get(key, 0) + value)
            for key, value in current.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", action="append", required=True)
    args = parser.parse_args()
    if len(args.source) < 2:
        parser.error("at least two --source runs are required")
    if not config.MODEL or not config.API_BASE or not config.API_KEY:
        raise SystemExit("configure CMEL_MODEL, CMEL_API_BASE and CMEL_API_KEY")

    source_dirs = [Path(config.RUNS_DIR) / name for name in args.source]
    selections = [read_json(path / "selection.json") for path in source_dirs]
    if any(selection != selections[0] for selection in selections[1:]):
        raise SystemExit("source runs use different document selections")
    _, methods_module = llm_backend.install(mode="api")
    del methods_module
    import methods

    out_dir = Path(config.RUNS_DIR) / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": "ensemble_candidate_adjudication_v1",
        "source_runs": args.source,
        "text_model": config.MODEL,
        "text_api_base": config.API_BASE,
        "generation": config.generation_config(),
        "text_cache_namespace": config.TEXT_CACHE_NAMESPACE,
        "pipeline_schema": config.PIPELINE_SCHEMA_VERSION,
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists() and read_json(config_path) != run_config:
        raise SystemExit(f"run {args.run!r} exists with a different configuration")
    write_json_atomic(str(config_path), run_config)
    write_json_atomic(str(out_dir / "selection.json"), selections[0])

    started = time.time()
    completed_this_process = 0
    stats_path = out_dir / "llm_stats.json"
    progress_path = out_dir / "progress.json"
    prior_stats = read_json(stats_path) if stats_path.exists() else {}
    prior_elapsed = (float(read_json(progress_path).get("elapsed_s", 0))
                     if progress_path.exists() else 0.0)
    images_expected = sum(
        len(read_json(source_dirs[0] / category / document / "result.json"))
        for category, documents in selections[0].items() for document in documents
    )
    for category, documents in selections[0].items():
        for document in documents:
            source_results = [read_json(path / category / document / "result.json")
                              for path in source_dirs]
            doc_data = data.load_doc(category, document)
            image_ids = list(source_results[0])
            if any(set(result) != set(image_ids) for result in source_results[1:]):
                raise SystemExit(f"image keys differ for {category}/{document}")
            doc_dir = out_dir / category / document
            doc_dir.mkdir(parents=True, exist_ok=True)
            result_path = doc_dir / "result.json"
            output = read_json(result_path) if result_path.exists() else {}
            for image_id in image_ids:
                if image_id in output:
                    continue
                candidates = union_candidates(source_results, image_id, args.source)
                if candidates:
                    chunk_index = doc_data["image_data"][image_id].get("chunk_order_index")
                    image_entities = doc_data["image_kg"].get(image_id, [])
                    text_entities = methods.get_nearby_entities(doc_data["chunk_kg"], chunk_index)
                    chunks = methods.get_nearby_chunks(doc_data["text_chunks"], chunk_index)
                    prompt = build_prompt(candidates, image_entities, text_entities, chunks)
                    allowed = {item["candidate_id"] for item in candidates}
                    keep = set(parse_keep_ids(
                        llm_backend.get_llm_response(prompt, SYSTEM_PROMPT), allowed))
                    merged = [item["_record"] for item in candidates
                              if item["candidate_id"] in keep]
                else:
                    merged = []
                primary = source_results[0].get(image_id) or {}
                output[image_id] = {
                    "matched_entity": primary.get("matched_entity", "no match"),
                    "merged_entities": merged,
                }
                write_json_atomic(str(result_path), output)
                completed_this_process += 1
            results = evaluate.evaluate_run(str(out_dir))
            cumulative_stats = sum_stats(prior_stats, llm_backend.stats())
            write_json_atomic(str(out_dir / "metrics.json"), results)
            write_json_atomic(str(stats_path), cumulative_stats)
            write_json_atomic(str(progress_path), {
                "status": "running", "last_document": f"{category}/{document}",
                "images_completed": count_checkpointed_images(out_dir),
                "images_expected": images_expected,
                "images_completed_this_process": completed_this_process,
                "elapsed_s": round(prior_elapsed + time.time() - started, 3),
                "llm": cumulative_stats,
            })
            print(f"[{category}/{document}] images={len(image_ids)} "
                  f"new_total={completed_this_process} "
                  f"elapsed={time.time()-started:.0f}s")

    results = evaluate.evaluate_run(str(out_dir))
    analysis = analyze_task2_errors.analyze_run(out_dir)
    write_json_atomic(str(out_dir / "metrics.json"), results)
    cumulative_stats = sum_stats(prior_stats, llm_backend.stats())
    write_json_atomic(str(stats_path), cumulative_stats)
    write_json_atomic(str(out_dir / "task2_error_analysis.json"), {"primary": analysis})
    (out_dir / "task2_error_analysis.md").write_text(
        analyze_task2_errors.markdown({"primary": analysis}), encoding="utf-8")
    (out_dir / "report.md").write_text(make_report(out_dir, results, analysis), encoding="utf-8")
    write_json_atomic(str(progress_path), {
        "status": "complete", "images_completed": count_checkpointed_images(out_dir),
        "images_expected": images_expected,
        "images_completed_this_process": completed_this_process,
        "elapsed_s": round(prior_elapsed + time.time() - started, 3),
        "llm": cumulative_stats,
    })
    print(make_report(out_dir, results, analysis))


if __name__ == "__main__":
    main()
