# CMEL vendor patches

实验直接导入 `../vendor/cmel_research/`；该目录是运行时真源，原始
`/private/mmkg/CMEL-dataset-main/cmel_research/` 保持不变。

补丁用途：

- `spectral_eigh.diff`：两处 SpecLink 拉普拉斯矩阵显式对称化并使用 `eigh`。
- `clustering_function.diff`：包含上述谱分解修复及 DBSCAN+KNN 分支修复。
- `pull_llm.diff`：用 JSON decoder 取代不支持嵌套结构的正则解析。
- `methods.diff`：规范化模型可能返回的 JSON 字符串/带引号实体名。

此外，`vendor/cmel_research/clustering_function.py` 还包含 LLM 数字簇标签校验、
最近邻降级和畸形融合项过滤；完整改动以 vendor 源文件及 `CHANGELOG.md` 为准。
`pull_llm.diff` 与 `methods.diff` 已使用 `patch --dry-run -p1` 验证可应用。
