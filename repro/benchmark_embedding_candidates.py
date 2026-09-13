"""Evaluate only MMGraphRAG's embedding candidate-generation stage.

This intentionally bypasses the later LLM/VLM judge so that an offline run can
measure differences between embedding models instead of measuring the stub.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

import config
import data


def norm_name(value):
    return str(value or "").strip().strip('"').replace("\\", "").casefold()


def nearby_entities(chunk_kg, index):
    found = []
    for i in range(max(0, index - 1), index + 2):
        found.extend(chunk_kg.get(str(i), {}).get("entities", []))
    return found


def add_task1(samples, doc_data, gt1, category, document):
    for image_id, gold in gt1.items():
        target = norm_name(gold.get("matched_chunk_entity_name"))
        if not target or target in {"no match", "nomatch"}:
            continue
        meta = doc_data["image_data"].get(image_id, {})
        candidates = nearby_entities(doc_data["chunk_kg"], meta.get("chunk_order_index", 0))
        samples.append({
            "task": "task1",
            "category": category,
            "document": document,
            "image_id": image_id,
            "query_name": gold.get("entity_name", image_id),
            "query": gold.get("description", ""),
            "gold": [target],
            "candidates": candidates,
        })


def add_task2(samples, doc_data, gt2, category, document):
    for image_id, alignments in gt2.items():
        image_entities = {
            norm_name(entity.get("entity_name")): entity
            for entity in doc_data["image_kg"].get(image_id, [])
        }
        meta = doc_data["image_data"].get(image_id, {})
        candidates = nearby_entities(doc_data["chunk_kg"], meta.get("chunk_order_index", 0))
        for alignment in alignments:
            gold = [norm_name(x) for x in alignment.get("source_text_entities", []) if norm_name(x)]
            for source_name in alignment.get("source_image_entities", []):
                source = image_entities.get(norm_name(source_name))
                if not source or not gold:
                    continue
                samples.append({
                    "task": "task2",
                    "category": category,
                    "document": document,
                    "image_id": image_id,
                    "query_name": source_name,
                    "query": source.get("description", ""),
                    "gold": gold,
                    "candidates": candidates,
                })


def collect_samples(selection):
    samples = []
    for category, documents in selection.items():
        for document in documents:
            doc_data = data.load_doc(category, document)
            gt1, gt2 = data.load_gt(category, document)
            add_task1(samples, doc_data, gt1, category, document)
            add_task2(samples, doc_data, gt2, category, document)
    return samples


def encode(model, texts, batch_size, prompt_name=None):
    kwargs = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "convert_to_numpy": True,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    return model.encode(texts, **kwargs)


def safe_div(num, den):
    return float(num / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selection", default=os.path.join(config.RUNS_DIR, "emb_minilm_actual", "selection.json"))
    ap.add_argument("--threshold", type=float, default=config.EMBEDDING_THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--query-prompt", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stella-shim", action="store_true")
    args = ap.parse_args()

    if args.stella_shim:
        from transformers import Qwen2Config
        Qwen2Config.rope_theta = property(
            lambda self: self.__dict__.get("rope_parameters", {}).get("rope_theta", 1_000_000.0)
        )

    from sentence_transformers import SentenceTransformer
    extra = {"config_kwargs": {"use_cache": False}} if args.stella_shim else {}
    model = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
        local_files_only=True,
        **extra,
    )

    with open(args.selection, "r", encoding="utf-8") as f:
        selection = json.load(f)
    samples = collect_samples(selection)

    query_texts = list(dict.fromkeys(s["query"] for s in samples))
    corpus_texts = list(dict.fromkeys(
        entity.get("description", "")
        for sample in samples
        for entity in sample["candidates"]
    ))
    query_vectors = encode(model, query_texts, args.batch_size, args.query_prompt)
    corpus_vectors = encode(model, corpus_texts, args.batch_size)
    qvec = dict(zip(query_texts, query_vectors))
    cvec = dict(zip(corpus_texts, corpus_vectors))

    counters = defaultdict(lambda: defaultdict(float))
    examples = []
    for sample in samples:
        candidates = sample["candidates"]
        key_pairs = [(sample["task"], "overall"), (sample["task"], sample["category"])]
        for key in key_pairs:
            counters[key]["n"] += 1
        if not candidates:
            continue
        scores = np.array([
            float(np.dot(qvec[sample["query"]], cvec[e.get("description", "")]))
            for e in candidates
        ])
        order = np.argsort(-scores)
        ranked = [(norm_name(candidates[i].get("entity_name")), float(scores[i])) for i in order]
        gold = set(sample["gold"])
        oracle = any(name in gold for name, _ in ranked)
        rank = next((i + 1 for i, (name, _) in enumerate(ranked) if name in gold), None)
        top1 = bool(rank and rank <= 1)
        top3 = bool(rank and rank <= 3)
        threshold_top3 = any(name in gold and score >= args.threshold for name, score in ranked[:3])
        for key in key_pairs:
            counters[key]["candidate_oracle"] += oracle
            counters[key]["recall_at_1"] += top1
            counters[key]["recall_at_3"] += top3
            counters[key]["threshold_recall_at_3"] += threshold_top3
            counters[key]["mrr"] += (1.0 / rank) if rank else 0.0
        if len(examples) < 30:
            examples.append({
                **{k: sample[k] for k in ("task", "category", "document", "image_id", "query_name")},
                "gold": sample["gold"],
                "top3": [{"entity": n, "score": round(s, 4)} for n, s in ranked[:3]],
                "gold_rank": rank,
            })

    metrics = {}
    for (task, category), values in sorted(counters.items()):
        n = int(values["n"])
        metrics.setdefault(task, {})[category] = {
            "n": n,
            "candidate_oracle_pct": round(100 * safe_div(values["candidate_oracle"], n), 2),
            "recall_at_1_pct": round(100 * safe_div(values["recall_at_1"], n), 2),
            "recall_at_3_pct": round(100 * safe_div(values["recall_at_3"], n), 2),
            "threshold_recall_at_3_pct": round(100 * safe_div(values["threshold_recall_at_3"], n), 2),
            "mrr": round(safe_div(values["mrr"], n), 4),
        }

    result = {
        "model": args.model,
        "device": args.device,
        "embedding_dimension": int(query_vectors.shape[1]),
        "threshold": args.threshold,
        "query_prompt": args.query_prompt,
        "selection": selection,
        "metrics": metrics,
        "examples": examples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
