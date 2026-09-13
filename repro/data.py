"""数据集加载。不修改原始数据集，image_path 在内存里重写成本机绝对路径。"""
import json
import os
import random

import config

FILES = {
    "image_data": "kv_store_image_data.json",
    "text_chunks": "kv_store_text_chunks.json",
    "chunk_kg": "kv_store_chunk_knowledge_graph.json",
    "image_kg": "kv_store_image_knowledge_graph.json",
}
GT_TASK1 = "aligned_image_entity.json"   # task1 / task3 的标注
GT_TASK2 = "aligned_text_entity.json"    # task2 的标注（论文 Table 1 的 1114 条）


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def doc_dir(cat, doc):
    return os.path.join(config.DATASET_DIR, cat, doc)


def list_docs(cat):
    d = os.path.join(config.DATASET_DIR, cat)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def task2_gt_count(cat, doc):
    gt = _read(os.path.join(doc_dir(cat, doc), GT_TASK2))
    return sum(len(v) for v in gt.values())


def select_docs(n_per_cat, seed=0, require_task2=True):
    """确定性抽样：固定 seed，保证多次运行/多种方法用的是同一批文档。"""
    out = {}
    for cat in config.CATEGORIES:
        docs = [d for d in list_docs(cat)
                if not require_task2 or task2_gt_count(cat, d) > 0]
        rng = random.Random(seed)
        out[cat] = sorted(rng.sample(docs, min(n_per_cat, len(docs))))
    return out


def load_doc(cat, doc):
    base = doc_dir(cat, doc)
    data = {k: _read(os.path.join(base, fn)) for k, fn in FILES.items()}

    # 重写 image_path：数据集里存的是作者机器上的相对前缀
    for _, meta in data["image_data"].items():
        p = meta.get("image_path", "")
        if p.startswith(config.DATASET_PATH_PREFIX):
            rel = p[len(config.DATASET_PATH_PREFIX):].lstrip("/\\")
            meta["image_path"] = os.path.join(config.DATASET_DIR, rel)
        elif not os.path.isabs(p):
            meta["image_path"] = os.path.join(base, os.path.basename(os.path.dirname(p)),
                                              os.path.basename(p))
    return data


def check_images(cat, doc):
    """返回 (缺失数, 总数)，用于跑之前确认图片路径重写正确。"""
    data = load_doc(cat, doc)
    paths = [m["image_path"] for m in data["image_data"].values()]
    return sum(1 for p in paths if not os.path.exists(p)), len(paths)


def load_gt(cat, doc):
    base = doc_dir(cat, doc)
    return _read(os.path.join(base, GT_TASK1)), _read(os.path.join(base, GT_TASK2))
