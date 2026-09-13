"""CMEL 复现主入口。

对应原仓库 cmel_research/fusion_research.py，但：
  - 结果写到独立 run 目录，不污染数据集
  - 断点续跑到「单张图片」粒度，而不是整篇文档
  - 单张图片失败不会中断整篇文档
  - LLM 调用带磁盘缓存，重跑同一配置几乎零成本

用法:
  python run_cmel.py --method clustering --clustering spectral --classify knn \\
                     --docs 4 --run spec_mini --mode stub
"""
import argparse
import json
import os
import sys
import time
import traceback

import config
import data
import llm_backend

METHODS = ["embedding", "llm", "clustering", "task3"]
CLUSTERINGS = ["spectral", "dbscan", "kmeans", "pagerank", "leiden"]


def write_json_atomic(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json_if_present(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sum_stats(previous, current):
    return {
        key: round(previous.get(key, 0) + value, 3)
        if isinstance(value, float) else previous.get(key, 0) + value
        for key, value in current.items()
    }


def _count_checkpointed_images(run_dir):
    count = 0
    for root, _, files in os.walk(run_dir):
        if "result.json" not in files:
            continue
        try:
            count += len(_read_json_if_present(os.path.join(root, "result.json"), {}))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return count


def save_run_snapshot(run_dir, elapsed, totals, last_document,
                      prior_stats, prior_elapsed, expected_images):
    """每篇文档完成后保存状态和增量评测；中断后这些文件仍可直接阅读。"""
    stats = _sum_stats(prior_stats, llm_backend.stats())
    write_json_atomic(os.path.join(run_dir, "llm_stats.json"), stats)
    write_json_atomic(os.path.join(run_dir, "progress.json"), {
        "status": "running",
        "last_document": last_document,
        "elapsed_s": round(prior_elapsed + elapsed, 3),
        "images_expected": expected_images,
        "images_completed": _count_checkpointed_images(run_dir),
        "images_total_seen_this_process": totals[0],
        "images_completed_this_process": totals[1],
        "images_failed_this_process": totals[2],
        "llm": stats,
        "updated_at_unix": time.time(),
    })

    import evaluate
    results = evaluate.evaluate_run(run_dir)
    write_json_atomic(os.path.join(run_dir, "metrics.json"), results)
    report_path = os.path.join(run_dir, "report.md")
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(evaluate.format_report(run_dir, results))
    os.replace(tmp, report_path)


def build_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE,
                               trust_remote_code=True)


def run_one_image(methods, method, args, model, image_id, d):
    """返回写进 result.json 的那条记录。"""
    if method == "task3":
        aligned = methods.align_single_image_entity(image_id, d["image_data"], d["text_chunks"])
        return aligned if aligned else "no match"

    if method == "embedding":
        matched, merged = methods.entity_alignment_embedding(
            image_id, d["image_data"], d["text_chunks"], d["chunk_kg"], d["image_kg"],
            model, config.EMBEDDING_THRESHOLD)
    elif method == "llm":
        matched, merged = methods.entity_alignment_llm(
            image_id, d["image_data"], d["text_chunks"], d["chunk_kg"], d["image_kg"])
    elif method == "clustering":
        matched, merged = methods.entity_alignment_clustering(
            model, args.clustering, args.classify, image_id,
            d["image_data"], d["text_chunks"], d["chunk_kg"], d["image_kg"])
    else:
        raise ValueError(method)
    return {"matched_entity": matched, "merged_entities": merged}


def run_doc(methods, method, args, model, cat, doc, run_dir):
    out_dir = os.path.join(run_dir, cat, doc)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")

    result = {}
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

    d = data.load_doc(cat, doc)
    image_ids = list(d["image_data"].keys())
    todo = [i for i in image_ids if i not in result]
    if not todo:
        return len(image_ids), 0, 0

    failed = 0
    for image_id in todo:
        try:
            result[image_id] = run_one_image(methods, method, args, model, image_id, d)
        except Exception:
            failed += 1
            print(f"  [FAIL] {cat}/{doc}/{image_id}", file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)
            continue
        # 每张图片落盘一次：中断后可以精确续跑
        tmp = result_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, result_path)
    return len(image_ids), len(todo) - failed, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--clustering", choices=CLUSTERINGS, default="spectral")
    ap.add_argument("--classify", choices=["knn", "llm"], default="knn")
    ap.add_argument("--docs", type=int, default=4, help="每个域抽多少篇文档")
    ap.add_argument("--all-docs", action="store_true", help="使用所有含 Task 2 标注的文档")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", required=True, help="run 名称，结果写到 runs/<run>/")
    ap.add_argument("--mode", choices=["api", "stub"], default="api")
    ap.add_argument("--categories", nargs="*", default=config.CATEGORIES)
    args = ap.parse_args()

    if args.mode == "api" and not config.api_configured():
        sys.exit("API 未配置。设置 CMEL_API_KEY / CMEL_API_BASE / CMEL_MODEL / "
                 "CMEL_MM_MODEL 环境变量，或用 --mode stub 做离线流水线验证。")

    methods, _ = llm_backend.install(mode=args.mode)

    docs_per_category = 10 ** 9 if args.all_docs else args.docs
    selection = data.select_docs(docs_per_category, seed=args.seed)
    missing_images = []
    for cat in args.categories:
        for doc in selection.get(cat, []):
            missing, total = data.check_images(cat, doc)
            if missing:
                missing_images.append(f"{cat}/{doc}: {missing}/{total}")
    if missing_images:
        sys.exit("图片文件缺失：\n" + "\n".join(missing_images))

    model = None
    if args.method in ("embedding", "clustering"):
        print(f"加载 embedding 模型 {config.EMBED_MODEL_NAME} ({config.EMBED_DEVICE}) ...")
        model = build_embedder()

    run_dir = os.path.join(config.RUNS_DIR, args.run)
    os.makedirs(run_dir, exist_ok=True)
    prior_stats = _read_json_if_present(os.path.join(run_dir, "llm_stats.json"), {})
    prior_progress = _read_json_if_present(os.path.join(run_dir, "progress.json"), {})
    prior_elapsed = float(prior_progress.get("elapsed_s", 0))
    run_config = {
        **vars(args),
        "pipeline_schema": config.PIPELINE_SCHEMA_VERSION,
        "embed_model": config.EMBED_MODEL_NAME,
        "embed_device": config.EMBED_DEVICE,
        "embedding_threshold": config.EMBEDDING_THRESHOLD,
        "text_model": config.MODEL,
        "text_api_base": config.API_BASE,
        "mm_model": config.MM_MODEL,
        "mm_api_base": config.MM_API_BASE,
        "generation": config.generation_config(),
        "cache_namespace": config.CACHE_NAMESPACE,
        "text_cache_namespace": config.TEXT_CACHE_NAMESPACE,
        "mm_cache_namespace": config.MM_CACHE_NAMESPACE,
        "alignment_prompt_version": config.ALIGNMENT_PROMPT_VERSION,
        "dense_candidate_top_k": config.DENSE_CANDIDATE_TOP_K,
    }
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        # schema-4 早期 run 未显式记录提示词版本，等价于上游 paper_v1。
        previous.setdefault("alignment_prompt_version", "paper_v1")
        previous.setdefault("dense_candidate_top_k", 5)
        if previous != run_config:
            sys.exit(
                f"run {args.run!r} 已存在且配置不同；请使用新的 --run 名称，"
                "防止混合实验结果。"
            )
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)

    write_json_atomic(os.path.join(run_dir, "selection.json"), selection)
    expected_images = sum(
        data.check_images(cat, doc)[1]
        for cat in args.categories for doc in selection.get(cat, [])
    )

    t0 = time.time()
    tot_img = tot_ok = tot_fail = 0
    for cat in args.categories:
        for doc in selection.get(cat, []):
            n, ok, fail = run_doc(methods, args.method, args, model, cat, doc, run_dir)
            tot_img += n
            tot_ok += ok
            tot_fail += fail
            print(f"[{cat}/{doc}] images={n} new_ok={ok} failed={fail} "
                  f"elapsed={time.time()-t0:.0f}s")
            save_run_snapshot(
                run_dir,
                time.time() - t0,
                (tot_img, tot_ok, tot_fail),
                {"category": cat, "document": doc},
                prior_stats,
                prior_elapsed,
                expected_images,
            )

    print(f"\n完成: 图片总数={tot_img} 本次成功={tot_ok} 失败={tot_fail} "
          f"耗时={time.time()-t0:.0f}s")
    print("LLM 调用统计:", llm_backend.stats())
    print(f"结果目录: {run_dir}")
    progress_path = os.path.join(run_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        progress["status"] = (
            "complete" if progress.get("images_completed") == expected_images
            else "incomplete"
        )
        progress["updated_at_unix"] = time.time()
        write_json_atomic(progress_path, progress)


if __name__ == "__main__":
    main()
