# Change log

## 2026-09-11

- 集中配置本地数据、VLM/LLM、embedding、生成参数与缓存命名空间。
- 新增 OpenAI-compatible 后端：重试、原始响应审计、thinking/Markdown 清洗、
  schema-2 缓存键、token 与延迟统计。
- 主实验支持单图 checkpoint、逐文档原子保存、断点续跑、全量数据、配置冲突保护、
  缺图预检和自动 Markdown 报告。
- 修复 SpecLink 复数特征向量、DBSCAN+KNN 未定义标签、LLM 簇标签任意回退、
  嵌套 JSON 解析和引号包裹实体名等上游问题。
- 新增 oracle/null 评测器自检，并明确 Task 2 micro/macro 与论文完整集判据。
- 模型统一注册到 `/root/models`；分别保留 Qwen3.8-27B 的 vLLM 0.28 环境和
  Qwen3.8-Flash-Next-FP8 的 vLLM 0.29 环境。
- Flash 在 2 × L20X 上采用 TP=2、显存比例 0.85、禁用 inductor compile、
  decode-only CUDA graph；规避 23.84 GiB 编译期额外分配并显著提升吞吐。
- 完成 Qwen3.8-27B 与 Flash 的固定 dev12 对比；正在运行 Flash 完整集。
- 完成 Flash temperature=0 完整集基线：Task 2 48.1/55.4，未超过论文；保留为
  不覆盖的基线 run。
- pipeline schema 升至 3：标准 decoder 失败时使用 json-repair 0.63.4；新 run 的
  `run_config.json` 显式记录 schema，防止解析版本混淆。
- pipeline schema 升至 4：支持 `CMEL_TEXT_CACHE_NAMESPACE` 与
  `CMEL_MM_CACHE_NAMESPACE`，使 Flash-VLM + 27B-LLM 能在不同时驻留的前提下复用
  各自可审计缓存。
- 修复 `run_speclink.sh` 无条件覆盖 `CMEL_MODEL`/`CMEL_MM_MODEL` 的问题；误配置 run
  已写入 `INVALID_RUN.md`，不纳入结果比较。
- 完成 Flash 官方 non-thinking 采样、thinking-low mini、schema-3 JSON 修复复算及
  Flash-VLM + 27B-LLM 消融；均未优于 Flash temperature=0 dev12 基线。
- 有效混合 run 已由独立 text/mm cache namespace 审计：Task 1 为 58.8/56.4，Task 2
  为 53.1/58.0；后续优化焦点转为细粒度候选与文本判定链路。
- 新增 `analyze_task2_errors.py`，持久化严格匹配错误桶、最差文档、典型样例与跨 run
  oracle 互补性；Flash/27B dev12 oracle 为 67.9%，明确标为不可报告诊断值。
- 新增可切换且写入 run 配置的 `alignment_prompt_version` 与 dense candidate Top-K。
  `strict_v2` dev12（42.3/45.9）和 `hybrid_verify_v3` mini（19.5/8.2）均明显退化，
  已在扩大实验前淘汰；默认 `paper_v1` 完全保留。
- 启动 Qwen3.8-27B 的 1,114 条完整集正式对照，run 为
  `qwen38_27b_spec_llm_full_t0_schema4`。
- 新增 `build_union_ensemble.py` 协议审计：Flash∪27B dev12 的论文 accuracy 为
  67.9/72.5，但 precision 只有 33.2%，因此明确禁止将并集分数单独表述为真实提升。
- 新增带逐图 checkpoint 的 `adjudicate_ensemble.py`；27B 只能筛选已有候选，不能改写
  集合。dev12 达到 66.3/70.0，且链接 F1 从 Flash 的 46.6% 提至 48.1%，通过 full 门槛。
- 修正报告适用任务：聚类与集成方法不再把缺少 Task3 输出字段误报成 Task3=0；只有
  `--method task3` 的 run 才报告 Task3，oracle/null 自检仍覆盖全部任务。
- 完成 Qwen3.8-27B 全量基线：Task2 48.9/51.8，678/678 图、0 失败、17,168,199
  tokens、4,033 秒；单模型未超过论文。
- 完成完整集并集协议审计：59.1/63.6，但 precision=30.6%、F1=40.3%，不作为单独
  真实提升结论。
- 使用 dev12 冻结的 `ensemble_candidate_adjudication_v1` 完成全量：Task2
  **56.3/61.3**，相对 Spe-L **+4.5/+2.1 pp**，n=1,114、missing=0；链接 F1=41.7%。
  主目标已达成，全部源码、配置、逐图结果、缓存、统计与 Markdown 报告已落盘。
