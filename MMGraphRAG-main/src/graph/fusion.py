"""
知识图谱融合模块

将文本知识图谱与图像知识图谱进行对齐和融合。

主要流程:
1. 图像实体对齐 - 将图像实体与文本实体匹配
2. 图像知识图谱增强 - 使用文本信息增强图像图谱
3. 图谱融合 - 合并图像和文本知识图谱
"""
import math
import os
# 设置TOKENIZERS_PARALLELISM环境变量以禁用警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..core.base import logger, load_json, ensure_quoted
from ..core.prompt import PROMPTS, GRAPH_FIELD_SEP
from ..llm import get_llm_response, get_mmllm_response, normalize_to_json, normalize_to_json_list
from ..parameter import EMBED_MODEL, WORKING_DIR
from .. import parameter


# ============================================================================
# 数据加载函数
# ============================================================================

def _get_json_path(filename: str) -> str:
    """获取工作目录下的JSON文件路径"""
    return os.path.join(parameter.WORKING_DIR, filename)


def get_image_data() -> dict:
    """加载图像数据"""
    return load_json(_get_json_path("kv_store_image_data.json")) or {}


def get_chunk_knowledge_graph() -> dict:
    """加载文本块知识图谱"""
    return load_json(_get_json_path("kv_store_chunk_knowledge_graph.json")) or {}


def get_text_chunks() -> dict:
    """加载文本块"""
    return load_json(_get_json_path("kv_store_text_chunks.json")) or {}


# ============================================================================
# 上下文获取函数
# ============================================================================

def get_nearby_chunks(data: dict, index: int) -> list[str]:
    """获取指定索引附近的文本块内容"""
    indices = range(max(0, index - 1), min(len(data), index + 2))
    return [
        v.get("content") for v in data.values()
        if v.get("chunk_order_index") in indices
    ]


def get_nearby_entities(data: dict, index: int) -> list[dict]:
    """获取指定索引附近的实体列表"""
    indices = range(max(0, index - 1), min(len(data), index + 2))
    entities = []
    for i in indices:
        chunk_data = data.get(str(i), {})
        for entity in chunk_data.get("entities", []):
            # 复制并移除source_id
            entity_copy = {k: v for k, v in entity.items() if k != "source_id"}
            entities.append(entity_copy)
    return entities


def get_nearby_relationships(data: dict, index: int) -> list[dict]:
    """获取指定索引附近的关系列表"""
    indices = range(max(0, index - 1), min(len(data), index + 2))
    relationships = []
    for i in indices:
        chunk_data = data.get(str(i), {})
        for rel in chunk_data.get("relationships", []):
            rel_copy = {k: v for k, v in rel.items() if k != "source_id"}
            relationships.append(rel_copy)
    return relationships


def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    清洗和标准化嵌入矩阵，防止sklearn计算错误。
    1. 替换非法值 (NaN, Inf)
    2. 归一化 (Handling zero vectors)
    """
    if embeddings.size == 0:
        return embeddings

    # 1. 初始替换：替换 NaN 和 Inf
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 2. 数值裁剪：限制嵌入值在合理范围内，避免极大值导致溢出
    embeddings = np.clip(embeddings, -10.0, 10.0)  # 根据嵌入模型的特性调整阈值
    
    # 3. 检查并处理零向量
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_mask = (norms < 1e-8)
    
    if zero_mask.any():
        logger.warning(f"⚠️ 发现 {zero_mask.sum()} 个零/接近零向量嵌入")
        # 赋予零向量一个微小的随机值以避免除零
        embeddings[zero_mask.flatten()] = np.random.normal(0, 1e-6, 
            size=(zero_mask.sum(), embeddings.shape[1]))
        # 重新计算范数
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 4. 安全归一化
    # 使用 sklearn 的 normalize 方法可以保证数值稳定性
    normalized = embeddings / np.maximum(norms, 1e-8)
    
    # 5. 最终验证 (并强制转换为 float64 以提高 matmul 精度)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    return normalized.astype(np.float64)

# ============================================================================
# 谱聚类核心函数
# ============================================================================

def _compute_spectral_labels(
    embeddings: np.ndarray,
    entity_names: list[str],
    relationships: list[dict]
) -> list[int]:
    """
    使用谱聚类计算实体标签。
    
    Args:
        embeddings: 实体嵌入矩阵
        entity_names: 实体名称列表
        relationships: 关系列表（用于调整相似度矩阵）
    
    Returns:
        聚类标签列表
    """
    # 计算余弦相似度矩阵 (范围 [-1, 1])
    # 手动计算 cosine similarity 以避开 sklearn 的 bug
    normalized_embeddings = _sanitize_embeddings(embeddings)
    # 既然已经归一化，dot product 就是 cosine similarity
    raw_similarity = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # 1. 映射到 [0, 1] 区间，保证非负性 (Spectral Clustering 要求)
    similarity_matrix = (raw_similarity + 1.0) / 2.0
    
    # 清理相似度矩阵中的 NaN 和 Inf
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 根据关系权重调整相似度
    relationships = sorted(relationships, key=lambda x: x.get('weight', 0), reverse=True)
    for rel in relationships:
        src, tgt = rel.get("src_id"), rel.get("tgt_id")
        if src not in entity_names or tgt not in entity_names:
            continue
        
        weight_raw = rel.get("weight")
        if weight_raw is None:
            continue

        try:
            # 2. 权重逻辑：加法增强
            # 原始权重假设 1-100，将其映射到 [0, 0.5] 的相似度增益
            w_val = float(weight_raw)
            boost = min(w_val, 50.0) / 100.0  # max 0.5 boost
        except (ValueError, TypeError):
            continue
        
        if not np.isfinite(boost):
            continue
        
        src_idx = entity_names.index(src)
        tgt_idx = entity_names.index(tgt)
        
        # 加法增强：有关系的实体更相似
        similarity_matrix[src_idx, tgt_idx] += boost
        similarity_matrix[tgt_idx, src_idx] += boost
    
    # 再次截断到 [0, 1] (尽管加法可能稍微溢出，但限制一下更安全，或者允许 >1 也可以)
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.5)
    
    # 强制对称化（数值稳定性）
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0
    
    # 计算拉普拉斯矩阵
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    degree_matrix = np.clip(degree_matrix, 0, 1e6)  # 限制度矩阵的最大值
    laplacian_matrix = degree_matrix - similarity_matrix
    laplacian_matrix = np.nan_to_num(laplacian_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 对称化拉普拉斯矩阵（进一步确保稳定性）
    laplacian_matrix = (laplacian_matrix + laplacian_matrix.T) / 2.0
    
    # 特征分解 (使用 eigh 针对对称/Hermitian矩阵)
    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian_matrix)
    except np.linalg.LinAlgError:
        logger.warning("⚠️ 拉普拉斯矩阵特征分解失败，回退到普通eig")
        eigvals, eigvecs = np.linalg.eig(laplacian_matrix)
        
    # 取实部并排序（从小到大）
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    
    # 排序
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 选择前k个特征向量 (跳过第一个常数特征向量 if connected)
    k = max(2, math.ceil(math.sqrt(len(entity_names))))
    eigvecs_selected = eigvecs[:, :k]
    eigvecs_selected = _sanitize_embeddings(eigvecs_selected)
    
    # DBSCAN 聚类
    min_samples = max(1, math.ceil(len(entity_names) / 10))
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
    return dbscan.fit_predict(eigvecs_selected).tolist()

def _classify_by_nearest_neighbor(
    input_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    labels: list[int],
    n_neighbors: int = 1
) -> list[int]:
    """使用最近邻将输入嵌入分类到已有标签"""
    # 确保输入和参考嵌入都经过清洗
    input_embeddings = _sanitize_embeddings(input_embeddings)
    reference_embeddings = _sanitize_embeddings(reference_embeddings)
        
    # 手动实现最近邻搜索，避开 sklearn 的 bug
    # 计算余弦相似度: input (M, D) @ ref.T (D, N) -> (M, N)
    sims = np.dot(input_embeddings, reference_embeddings.T)
    
    # 确保相似度在 [-1, 1] 范围内 (处理浮点误差)
    sims = np.clip(sims, -1.0, 1.0)
    
    result_labels = []
    # 对每个输入找到最相似的 top-k
    for i in range(len(input_embeddings)):
        # argsort 返回从小到大的索引，取最后 n_neighbors 个，然后反转得到从大到小
        top_indices = np.argsort(sims[i])[-n_neighbors:][::-1]
        
        # 这里的逻辑是简单的 KNN 分类？原始代码似乎只取了第一个邻居的标签？
        # 原始代码: result_labels.append(labels[indices[0][0]])
        # 这里 indices[0][0] 是最近的一个邻居
        
        nearest_idx = top_indices[0]
        result_labels.append(labels[nearest_idx])
        
    return result_labels


# ============================================================================
# 实体对齐函数
# ============================================================================

def _prepare_and_cluster_entities(
    nearby_text_entities: list[dict],
    nearby_relationships: list[dict]
) -> tuple[np.ndarray, list[int]]:
    """
    Prepare embeddings and compute cluster labels for text entities.
    Returns: (embeddings, labels)
    """
    if not nearby_text_entities:
        return np.array([]), []
        
    descriptions = [e["description"] for e in nearby_text_entities]
    entity_names = [e["entity_name"] for e in nearby_text_entities]
    
    embeddings = EMBED_MODEL.encode(descriptions)
    embeddings = np.array(embeddings)
    embeddings = _sanitize_embeddings(embeddings)
    
    labels = _compute_spectral_labels(embeddings, entity_names, nearby_relationships)
    return embeddings, labels


def align_single_image_entity(img_entity_name: str, text_chunks: dict) -> dict:
    """对齐单个图像实体"""
    image_data = get_image_data()
    entity_info = image_data.get(img_entity_name, {})
    
    image_path = entity_info.get("image_path")
    description = entity_info.get("description", "")
    chunk_index = entity_info.get("chunk_order_index", 0)
    
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_index)
    entity_types = [t.upper() for t in PROMPTS["DEFAULT_ENTITY_TYPES"]]
    
    with open(image_path, "rb") as f:
        img_base = base64.b64encode(f.read()).decode("utf-8")
    
    prompt = PROMPTS["image_entity_alignment_user"].format(
        entity_type=entity_types,
        img_entity=img_entity_name,
        img_entity_description=description,
        chunk_text=nearby_chunks
    )
    result = get_mmllm_response(prompt, PROMPTS["image_entity_alignment_system"], img_base)
    return normalize_to_json(result)


def get_possible_entities_image_clustering(
    image_entity_description: str,
    nearby_text_entities: list[dict],
    nearby_relationships: list[dict]
) -> list[dict]:
    """通过聚类找到可能匹配的文本实体"""
    if not nearby_text_entities:
        return []
    
    # 获取聚类标签和嵌入
    embeddings, labels = _prepare_and_cluster_entities(nearby_text_entities, nearby_relationships)
    if embeddings.size == 0:
        return []
    
    # 分类图像实体
    input_embedding = EMBED_MODEL.encode([image_entity_description])
    target_label = _classify_by_nearest_neighbor(input_embedding, embeddings, labels, n_neighbors=3)[0]
    
    # 返回同一类别的实体
    return [e for e, label in zip(nearby_text_entities, labels) if label == target_label]


def get_possible_entities_text_clustering(
    filtered_image_entities: list[dict],
    nearby_text_entities: list[dict],
    nearby_relationships: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    对文本实体进行聚类，并将图像实体分配到对应类别。
    
    Returns:
        (image_entity_with_labels, text_clustering_results)
    """
    if not nearby_text_entities:
        return [], []
    
    # 获取聚类标签和嵌入
    embeddings, labels = _prepare_and_cluster_entities(nearby_text_entities, nearby_relationships)
    if embeddings.size == 0:
        return [], []
    
    # 分类图像实体
    image_entity_with_labels = []
    if filtered_image_entities:
        img_embeddings = EMBED_MODEL.encode([e["description"] for e in filtered_image_entities])
        img_labels = _classify_by_nearest_neighbor(img_embeddings, embeddings, labels)
        
        for entity, label in zip(filtered_image_entities, img_labels):
            image_entity_with_labels.append({
                "entity_name": entity["entity_name"],
                "label": label,
                "description": entity["description"],
                "entity_type": entity.get("entity_type", "image")
            })
    
    # 生成聚类结果
    text_clustering_results = []
    for label in set(labels):
        cluster_entities = [
            {
                "entity_name": e["entity_name"],
                "entity_type": e["entity_type"],
                "description": e["description"]
            }
            for e, l in zip(nearby_text_entities, labels) if l == label
        ]
        text_clustering_results.append({"label": label, "entities": cluster_entities})
    
    return image_entity_with_labels, text_clustering_results


def judge_image_entity_alignment(
    image_entity_name: str,
    image_entity_description: str,
    possible_entities: list[dict],
    nearby_chunks: list[str]
) -> str:
    """使用LLM判断图像实体与文本实体的匹配"""
    prompt = PROMPTS["image_entity_judgement_user"].format(
        img_entity=image_entity_name,
        img_entity_description=image_entity_description,
        possible_matched_entities=possible_entities,
        chunk_text=nearby_chunks
    )
    matched_entity_name = get_llm_response(prompt, PROMPTS["image_entity_judgement_system"])
    return matched_entity_name


def judge_text_entity_alignment_clustering(
    image_entity_with_labels: list[dict],
    text_clustering_results: list[dict]
) -> list[dict]:
    """使用LLM判断并融合实体"""
    clusters_info = []
    for cluster in text_clustering_results:
        clusters_info.append({
            "label": cluster["label"],
            "text_entities": [
                {
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "description": entity["description"],
                }
                for entity in cluster["entities"]
            ]
        })

    # 构建输入 prompt
    prompt_user = f"""
You are tasked with aligning image entities and text entities based on their labels and descriptions. Below are the clusters and the entities they contain.

Clusters information:
{{
    "clusters": [
        {", ".join([f'{{"label": {c["label"]}, "text_entities": {c["text_entities"]}}}' for c in clusters_info])}
    ]
}}

Image entities with labels:
{[
    {
        "entity_name": e["entity_name"],
        "label": e["label"],
        "description": e["description"],
        "entity_type": e["entity_type"]
    }
    for e in image_entity_with_labels
]}

Instruction:
1. For each image entity, look at the corresponding cluster (same label).
2. Compare the description and type of the image entity with the text entities in the same cluster.
3. Identify matching entities between the image entities and text entities within the same cluster (same label).
4. For each match, create a new unified entity by merging the descriptions and including the source entities under "source_image_entities" and "source_text_entities".
5. Output a JSON list where each item represents a merged entity with the following structure:
    {{
        "entity_name": "Newly merged entity name",
        "entity_type": "Type of the merged entity",
        "description": "Merged description of the entity",
        "source_image_entities": ["List of matched image entity names"],
        "source_text_entities": ["List of matched text entity names"]
    }}
Include only one JSON list as the output, strictly following the structure above.
"""
    prompt_system = """You are an AI assistant skilled in aligning entities based on semantic descriptions and cluster information. Use the provided instructions to merge entities accurately."""

    # 调用 LLM 获取融合结果
    merged_entities = get_llm_response(cur_prompt=prompt_user, system_content=prompt_system)
    normalized_merged_entities = normalize_to_json_list(merged_entities)
    return [
        item for item in normalized_merged_entities 
        if item.get("source_image_entities") and item.get("source_text_entities")
    ]


# ============================================================================
# 图像实体提取
# ============================================================================

def extract_image_entities(img_entity_name: str) -> list[dict]:
    """从GraphML文件提取图像实体"""
    path = os.path.join(
        parameter.WORKING_DIR, 
        f"images/{img_entity_name}/graph_{img_entity_name}_entity_relation.graphml"
    )
    
    if not os.path.exists(path):
        logger.warning(f"⚠️  未找到GraphML文件: {path}")
        return []
    
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    entities = []
    for node in root.findall('graphml:graph/graphml:node', ns):
        entity_name = (node.get('id') or "").strip('"')
        entity_type = "UNKNOWN"
        description = ""
        
        for data in node.findall('graphml:data', ns):
            key = data.get('key')
            text = (data.text or "").strip('"')
            if key == 'd0':
                entity_type = text
            elif key == 'd1':
                description = text
        
        entities.append({
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description
        })
    
    return entities


def enhance_image_entities(image_entities: list[dict], nearby_chunks: list[str]) -> list[dict]:
    """使用LLM增强图像实体描述"""
    prompt = PROMPTS["enhance_image_entity_user"].format(
        enhanced_image_entity_list=image_entities,
        chunk_text=nearby_chunks
    )
    result = get_llm_response(prompt, PROMPTS["enhance_image_entity_system"])
    return normalize_to_json_list(result)


# ============================================================================
# 知识图谱操作
# ============================================================================

def image_knowledge_graph_alignment(image_entity_name: str) -> list[dict]:
    """对齐图像知识图谱与文本实体"""
    image_data = get_image_data()
    chunk_kg = get_chunk_knowledge_graph()
    
    chunk_index = image_data[image_entity_name].get("chunk_order_index", 0)
    
    image_entities = extract_image_entities(image_entity_name)
    filtered = [e for e in image_entities if e['entity_type'] not in ["ORI_IMG", "IMG"]]
    
    nearby_entities = get_nearby_entities(chunk_kg, chunk_index)
    nearby_rels = get_nearby_relationships(chunk_kg, chunk_index)
    
    img_with_labels, text_clusters = get_possible_entities_text_clustering(
        filtered, nearby_entities, nearby_rels
    )
    
    return judge_text_entity_alignment_clustering(img_with_labels, text_clusters)


def enhanced_image_knowledge_graph(aligned_entities: list[dict], image_entity_name: str) -> str:
    """增强图像知识图谱"""
    image_data = get_image_data()
    text_chunks = get_text_chunks()
    
    img_kg_path = os.path.join(
        parameter.WORKING_DIR,
        f'images/{image_entity_name}/graph_{image_entity_name}_entity_relation.graphml'
    )
    enhanced_path = os.path.join(
        parameter.WORKING_DIR,
        f'images/{image_entity_name}/enhanced_graph_{image_entity_name}_entity_relation.graphml'
    )
    
    image_entities = extract_image_entities(image_entity_name)
    filtered = [e for e in image_entities if e['entity_type'] not in ["ORI_IMG", "IMG"]]
    
    chunk_index = image_data[image_entity_name].get("chunk_order_index", 0)
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_index)
    
    # 获取已对齐的图像实体
    aligned_image_names = []
    for entity in aligned_entities:
        src_imgs = entity.get('source_image_entities', [])
        if src_imgs:
            aligned_image_names.append(src_imgs[0])
    
    # 过滤出未对齐的实体进行增强
    to_enhance = [e for e in filtered if e['entity_name'] not in aligned_image_names]
    enhanced = enhance_image_entities(to_enhance, nearby_chunks)
    
    # 更新图谱
    G = nx.read_graphml(img_kg_path)
    
    for entity in enhanced:
        original_name = entity.get('original_name')
        if not original_name or 'description' not in entity:
            continue
        
        new_name = ensure_quoted(entity['entity_name'])
        
        for node_id in list(G.nodes()):
            if node_id.strip('"') == original_name:
                G = nx.relabel_nodes(G, {node_id: new_name})
                G.nodes[new_name]['description'] = entity['description']
                break
    
    nx.write_graphml(G, enhanced_path)
    return enhanced_path


def image_knowledge_graph_update(enhanced_path: str, image_entity_name: str) -> str:
    """更新图像知识图谱，添加与文本实体的连接"""
    image_data = get_image_data()
    text_chunks = get_text_chunks()
    chunk_kg = get_chunk_knowledge_graph()
    
    new_path = os.path.join(
        parameter.WORKING_DIR,
        f'images/{image_entity_name}/new_graph_{image_entity_name}_entity_relation.graphml'
    )
    
    image_entity = align_single_image_entity(image_entity_name, text_chunks)
    chunk_index = image_data[image_entity_name].get("chunk_order_index", 0)
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_index)
    nearby_entities = get_nearby_entities(chunk_kg, chunk_index)
    nearby_rels = get_nearby_relationships(chunk_kg, chunk_index)
    
    if not image_entity:
        return enhanced_path
    
    entity_name = image_entity.get("entity_name", "no_match")
    entity_desc = image_entity.get("description", "")
    
    if entity_name.lower().replace(" ", "") in ["no_match", "nomatch"]:
        return enhanced_path
    
    possible_matches = get_possible_entities_image_clustering(entity_desc, nearby_entities, nearby_rels)
    matched_name = judge_image_entity_alignment(entity_name, entity_desc, possible_matches, nearby_chunks)
    
    if not matched_name or not matched_name.strip():
        logger.warning(f"⚠️  未能匹配图像实体: {entity_name}")
        return enhanced_path
    
    matched_normalized = matched_name.strip().replace(" ", "").replace("\\", "").lower()
    
    G = nx.read_graphml(enhanced_path)
    
    # 查找ORI_IMG节点
    source_node = None
    for node, data in G.nodes(data=True):
        etype = data.get('entity_type', '')
        if etype in ['"ORI_IMG"', '"UNKNOWN"', 'ORI_IMG', 'UNKNOWN']:
            source_node = node
            break
    
    if source_node is None:
        logger.warning("未找到ORI_IMG节点")
        return enhanced_path
    
    # 获取边属性模板
    edges = list(G.edges(data=True))
    if edges:
        edge_data = edges[0][2]
        source_id = edge_data.get("source_id", "")
        order = edge_data.get("order", 1)
    else:
        source_id = G.nodes[source_node].get('source_id', '')
        order = 1
    
    # 查找匹配的文本实体
    matched = False
    for entity in nearby_entities:
        ename = entity.get("entity_name", "")
        if ename.strip().replace(" ", "").replace("\\", "").lower() == matched_normalized:
            matched = True
            quoted_name = ensure_quoted(ename)
            
            G.add_node(quoted_name,
                       entity_type=entity["entity_type"],
                       description=entity["description"],
                       source_id=source_id)
            
            G.add_edge(source_node, quoted_name,
                       weight=10.0,
                       description=f"{source_node} is the image of {ename}.",
                       source_id=source_id,
                       order=order)
            break
    
    if not matched:
        # 添加新的图像实体节点
        G.add_node(entity_name,
                   entity_type="IMG_ENTITY",
                   description=entity_desc,
                   source_id=source_id)
        
        G.add_edge(source_node, entity_name,
                   weight=10.0,
                   description=f"{source_node} is the image of {entity_name}.",
                   source_id=source_id,
                   order=order)
    
    nx.write_graphml(G, new_path)
    return new_path


def merge_graphs(
    image_graph_path: str,
    text_graph_path: str,
    aligned_entities: list[dict],
    image_entity_name: str
) -> str:
    """合并图像和文本知识图谱"""
    merged_path = os.path.join(parameter.WORKING_DIR, f'graph_merged_{image_entity_name}.graphml')
    
    image_graph = nx.read_graphml(image_graph_path)
    text_graph = nx.read_graphml(text_graph_path)
    
    if image_graph is None or text_graph is None:
        logger.error("❌ 加载图谱失败")
        return text_graph_path
    
    merged = nx.compose(image_graph, text_graph)
    
    for entity_info in aligned_entities:
        required_keys = ['entity_name', 'entity_type', 'description', 'source_image_entities', 'source_text_entities']
        if not all(k in entity_info for k in required_keys):
            continue
        
        src_image = entity_info['source_image_entities']
        src_text = entity_info['source_text_entities']
        
        if not src_image or not src_text:
            continue
        
        target = ensure_quoted(src_image[0])
        
        # 获取source_id
        src_id_img = image_graph.nodes.get(target, {}).get('source_id', '')
        src_id_txt = text_graph.nodes.get(ensure_quoted(src_text[0]), {}).get('source_id', '')
        combined_source_id = GRAPH_FIELD_SEP.join(filter(None, [src_id_img, src_id_txt]))
        
        # 合并实体
        all_entities = list(set(src_image + src_text))
        
        for entity in all_entities:
            entity = ensure_quoted(entity)
            if entity == target or entity not in merged.nodes:
                continue
                
            if entity in merged.nodes: 
                neighbors = list(merged.neighbors(entity))  # 获取当前实体的邻居
                for neighbor in neighbors:
                    if not merged.has_edge(target, neighbor):
                        merged.add_edge(target, neighbor)  # 将邻居与目标实体连接
                    # 将边的属性合并到目标节点的边
                    edge_data = merged.get_edge_data(entity, neighbor)
                    target_edge_data = merged.get_edge_data(target, neighbor)
                    
                    # 合并边的属性（传递所有属性，如 weight、description、source_id 和 order）
                    if target_edge_data:
                        # 如果边已经存在，合并现有的属性
                        for key in edge_data:
                            if key in ['weight', 'description', 'source_id', 'order']:
                                # 合并边的属性，如果已有就添加新值，或者保持已有的
                                target_edge_data[key] = edge_data.get(key, target_edge_data.get(key))
                    else:
                        # 如果边不存在，就添加新的边属性
                        merged[target][neighbor].update(edge_data)

                merged.remove_node(entity)  # 删除已经合并的实体节点
        
        # 更新目标节点属性
        if target not in merged.nodes:
            merged.add_node(target)
        
        merged.nodes[target].update({
            'entity_type': entity_info['entity_type'],
            'description': entity_info['description'],
            'source_id': combined_source_id
        })
        
        new_name = ensure_quoted(entity_info['entity_name'])
        if new_name != target:
            merged = nx.relabel_nodes(merged, {target: new_name})
    
    nx.write_graphml(merged, merged_path)
    logger.info(f"🔗 图谱融合完成: {merged_path}")
    return merged_path


# ============================================================================
# 主入口
# ============================================================================

async def fusion(img_ids: list[str]) -> str:
    """
    融合所有图像的知识图谱。
    
    Args:
        img_ids: 图像实体名称列表
    
    Returns:
        最终合并图谱的路径
    """
    graph_path = os.path.join(parameter.WORKING_DIR, 'graph_chunk_entity_relation.graphml')
    
    if not img_ids:
        return graph_path
    
    for image_name in tqdm(img_ids, desc="🔗 图谱融合", unit="张"):
        merged_path = os.path.join(parameter.WORKING_DIR, f'graph_merged_{image_name}.graphml')
        
        if os.path.exists(merged_path):
            graph_path = merged_path
            continue
        
        aligned = image_knowledge_graph_alignment(image_name)
        enhanced_path = enhanced_image_knowledge_graph(aligned, image_name)
        updated_path = image_knowledge_graph_update(enhanced_path, image_name)
        graph_path = merge_graphs(updated_path, graph_path, aligned, image_name)
    
    return graph_path