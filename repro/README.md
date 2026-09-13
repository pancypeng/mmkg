# MMGraphRAG / CMEL 可复现实验工作区

目标是在论文公开的 CMEL 协议上，以 `Stella + SpecLink + LLM` 为主线，比较
`Qwen3.8-27B` 和 `Qwen3.8-Flash-Next-FP8`，并超过论文 Spe-L 的完整集
Task 2 Overall **51.8% micro / 59.2% macro**。论文正文称 Table 1 是三个随机种子
的平均值，附录 A.3 又称完整表展示三次中的最好结果；本项目同时保存每次单独结果，
不掩盖这一口径差异。

当前冻结方法 `Flash/27B 候选并集 → 27B candidate-ID 裁决` 已在完整 77 篇、678 图、
1,114 条 Task2 gold 上达到 **56.3% micro / 61.3% macro**，相对论文
**+4.5/+2.1 pp**，缺失 0。正式 run 为
`runs/qwen38_flash_27b_adjudicated_by_27b_full_v1/`；完整过程和局限见
`EXPERIMENT_STATUS.md` 与 `MMGraphRAG_STEP_BY_STEP_REPORT.md`。

## 目录与持久化

```text
/private/mmkg/repro/
├── config.py                    # 路径、模型、生成参数、缓存命名空间
├── data.py                      # 数据加载、路径重写、确定性选样
├── llm_backend.py               # OpenAI 兼容调用、重试、隔离缓存、token/延迟统计
├── run_cmel.py                  # 主实验；单图 checkpoint、单文档刷新报告
├── evaluate.py                  # Task 1/2/3 的 micro/macro 统计
├── validate_evaluator.py        # oracle/null 评测器自检
├── analyze_task2_errors.py      # 错误桶、oracle 与 precision/recall/F1
├── build_union_ensemble.py      # 不调用模型的并集协议审计
├── adjudicate_ensemble.py       # 带 checkpoint 的候选二次裁决
├── scripts/
│   ├── serve_qwen.sh            # 两种 Qwen 的 TP=2 vLLM 启动入口
│   ├── preflight.py             # 模型、数据、图片、GPU、API 只读预检
│   ├── smoke_vlm.py             # 固定文本 + 真实图片 VLM smoke test
│   └── run_speclink.sh          # dev/full 统一实验入口
├── vendor/cmel_research/        # 实际执行的、已修复的上游源码副本
├── patches/                     # 相对上游发布版的补丁记录
├── cache/{text,mm}/             # 请求、原始响应、清洗响应、用量与延迟
└── runs/<run_name>/             # 每次实验的全部可恢复产物
```

长实验不会只存在于终端输出中。每张图片完成后更新对应 `result.json`；每篇文档完成后
原子更新 `progress.json`、`llm_stats.json`、`metrics.json` 和 `report.md`。

## 本机环境

- GPU：2 × NVIDIA L20X，单卡 143,771 MiB，Compute Capability 8.9，卡间 NVLink。
- CUDA driver：580.105.08；系统 `nvcc` 13.0。
- 推理环境：`/mnt/conda-store/envs/qwen-vllm`
  - Python 3.12、PyTorch 2.13.0+cu132、Transformers 5.16.1、vLLM 0.28.0。
- Flash 推理环境：`/mnt/conda-store/envs/qwen-flash-vllm`
  - 从上述环境克隆后仅升级至 vLLM 0.29.0；原 0.28 环境保留用于 27B 回滚。
- CMEL/Stella 环境：`/private/mmkg/.venv-stella310`
  - Python 3.10、PyTorch 2.3.1+cu121、Transformers 4.42.3、
    SentenceTransformers 3.0.1；已补充 OpenAI 3.13.0、igraph 1.0.0、
    leidenalg 0.12.0、json-repair 0.63.4。
  - 补充依赖锁定在 `requirements-eval.txt`。
- 数据：`/private/mmkg/CMEL-dataset-main/CMEL_dataset`
  - 全部 Task 2 标注为 1,114 条；有 Task 2 标注的 77 篇文档含 678 张图片；图片缺失 0。

统一模型目录与清单见 `/root/models/README.md` 和 `/root/models/models.json`：

| 角色 | 模型 | 本地路径 |
|---|---|---|
| VLM/LLM | Qwen3.8-27B | `/root/models/Qwen3.8-27B` |
| VLM/LLM | Qwen3.8-Flash-Next-FP8 | `/root/models/Qwen3.8-Flash-Next-FP8` |
| EMB | Stella-en-1.5B-v5 | `/root/models/stella_en_1.5B_v5` |
| EMB 消融 | BGE-M3 | `/root/models/bge-m3` |
| EMB 消融 | all-MiniLM-L6-v2 | `/root/models/all-MiniLM-L6-v2` |

## 1. 启动模型

在独立终端启动；同一时刻只驻留一个大模型：

```bash
cd /private/mmkg/repro
./scripts/serve_qwen.sh 27b
# 或停止 27B 后：
./scripts/serve_qwen.sh flash
```

固定服务参数：`tensor_parallel_size=2`、`max_model_len=32768`、
`max_num_seqs=8`、每请求最多 1 张图、开启 prefix caching。27B 默认
`gpu_memory_utilization=0.75`；Flash FP8 默认 `0.85`。服务名分别是
`qwen38-27b` 与 `qwen38-flash-next-fp8`。额外 vLLM 参数可追加在脚本末尾。
Flash 使用官方 recipe 要求的最低 vLLM 0.29，并额外设置
`--no-enable-flashinfer-autotune --moe-backend triton`、关闭 inductor compile 且仅捕获
decode CUDA graph；当前 PyPI 0.28
无法识别 `Qwen4ExpForConditionalGeneration`。在 L20X 上不加 `--enforce-eager` 时，
torch.compile/profile 会为 Ngram 测试张量额外申请约 23.84 GiB 并 OOM；上述参数已完成
文本、真实图片与完整预检。启动后每卡权重约 86.92 GiB、KV cache 约 28.19 GiB，
空闲约 20.4 GiB。decode-only graph 在同一 dev 流程中约为 5.3 秒/图，而完全 eager
约为 46 秒/图；这只改变执行图，不改变模型权重、提示词或生成参数。

如遇驱动或 graph 兼容问题，可回退到完全 eager：

```bash
FLASH_EXECUTION_MODE=eager ./scripts/serve_qwen.sh flash
```

## 2. 实验前检查

服务出现 `Application startup complete` 后执行：

```bash
cd /private/mmkg/repro
/private/mmkg/.venv-stella310/bin/python scripts/preflight.py
/private/mmkg/.venv-stella310/bin/python validate_evaluator.py 4
```

预检必须同时满足：两个 VLM 与 Stella 的 `config.json` 存在；Task 2 恰为 1,114 条；
所有图片存在；两张 GPU 可见；`http://127.0.0.1:8000/v1/models` 返回预期服务名。
评测器 oracle 应为 100%，null 应为 0%。

文本与真实图片 smoke test（模型名按当前服务替换）：

```bash
/private/mmkg/.venv-stella310/bin/python scripts/smoke_vlm.py \
  --model qwen38-flash-next-fp8 \
  --image /private/mmkg/CMEL-dataset-main/CMEL_dataset/news/20231013/images/image_1.jpg \
  --output runs/flash_smoke.json
```

## 3. 执行与恢复

先跑固定 12 篇开发子集（每域 4 篇、seed 0）：

```bash
./scripts/run_speclink.sh qwen38-27b qwen38_27b_spec_llm_dev12_t0 dev
./scripts/run_speclink.sh qwen38-flash-next-fp8 qwen38_flash_spec_llm_dev12_t0 dev
```

较昂贵的新推理配置可先用 `mini`（每域 1 篇，共 3 篇）预筛选；它是 dev12 的真子集：

```bash
./scripts/run_speclink.sh qwen38-flash-next-fp8 RUN_NAME mini
```

选定开发配置后跑完整 Task 2：

```bash
./scripts/run_speclink.sh qwen38-flash-next-fp8 qwen38_flash_spec_llm_full_t0 full
```

默认生成参数是适合结构化抽取的确定性基线：`temperature=0`、`top_p=1`、
`max_tokens=2048`、`seed=0`、关闭 thinking。所有参数均写入缓存键和
`run_config.json`。可用环境变量覆盖，例如 Qwen 官方 non-thinking 采样配置：

```bash
CMEL_LLM_TEMPERATURE=0.7 CMEL_LLM_TOP_P=0.8 CMEL_LLM_TOP_K=20 \
CMEL_LLM_PRESENCE_PENALTY=1.5 CMEL_LLM_SEED=1 \
CMEL_CACHE_NAMESPACE=qwen38-flash-official-seed1-v2 \
./scripts/run_speclink.sh qwen38-flash-next-fp8 qwen38_flash_dev_seed1 dev
```

上述是官方推荐采样参数的复现实例，不代表本任务的推荐配置。本机 dev12 实测
Task 2 只有 22.4%/29.4%，显著低于 temperature=0 的 56.1%/63.1%；结构化抽取实验
继续以 temperature=0 为基线。

恢复时必须使用完全相同的命令和环境变量。程序会跳过已有图片；若同一 run 名的配置
发生变化，会拒绝启动，防止拼接不同实验。需要新参数时使用新的 run 名和缓存命名空间。

混合模型实验可分别指定文本与视觉缓存。这样能先用 Flash 生成并缓存全部视觉实体，
再切换到 27B 服务做文本判定，不要求两个大模型同时驻留：

```bash
CMEL_MODEL=qwen38-27b CMEL_MM_MODEL=qwen38-flash-next-fp8 \
CMEL_TEXT_CACHE_NAMESPACE=qwen38-27b-nonthink-t0-v2 \
CMEL_MM_CACHE_NAMESPACE=qwen38-flash-next-fp8-nonthink-t0-v2 \
./scripts/run_speclink.sh qwen38-27b RUN_NAME dev
```

视觉缓存必须已覆盖所选全部图片；若出现 miss，而当前服务没有
`qwen38-flash-next-fp8`，请求会明确失败，不会静默改用 27B。

手工重新统计不会调用模型：

```bash
/private/mmkg/.venv-stella310/bin/python evaluate.py --run RUN_NAME
```

逐条 Task 2 误差分解与跨 run 互补性诊断：

```bash
/private/mmkg/.venv-stella310/bin/python analyze_task2_errors.py \
  --run PRIMARY_RUN --compare SECOND_RUN
```

命令会生成 `runs/PRIMARY_RUN/task2_error_analysis.json` 和 `.md`。跨 run oracle 仅回答
“是否存在互补上限”，不能当作模型分数或超越论文的证据。

两条同 selection 的预测可先做零模型调用的并集审计：

```bash
/private/mmkg/.venv-stella310/bin/python build_union_ensemble.py \
  --run UNION_RUN --source FLASH_RUN --source QWEN27_RUN
```

并集会利用论文 accuracy 不惩罚假阳性的性质，必须同时查看生成报告中的 precision/F1。
推荐的可执行集成是冻结候选后由文本模型二次裁决：

```bash
CMEL_API_KEY=local CMEL_API_BASE=http://127.0.0.1:8000/v1 \
CMEL_MODEL=qwen38-27b CMEL_TEXT_CACHE_NAMESPACE=qwen38-27b-ensemble-adjudicate-v1-t0 \
/private/mmkg/.venv-stella310/bin/python -u adjudicate_ensemble.py \
  --run ADJUDICATED_RUN --source FLASH_RUN --source QWEN27_RUN
```

裁决器只返回候选 ID，代码再复制原候选，因此模型不能创建、拆分或组合实体集合。每张图
完成后原子更新 `result.json`，每篇文档后更新 metrics/progress/stats；相同 run 可断点续跑。

## 4. 每个 run 的产物

- `run_config.json`：方法、模型、endpoint、EMB、生成参数、缓存命名空间。
- `selection.json`：实际文档清单；不同模型可核对是否完全一致。
- `<domain>/<document>/result.json`：逐图片最终输出，也是恢复 checkpoint。
- `progress.json`：运行状态、最后完成文档、成功/失败图片数、耗时。
- `llm_stats.json`：命中/未命中、文本/多模态调用数、token、API 累计延迟。
- `metrics.json`：机器可读的 micro/macro、样本数、文档数和缺失数。
- `report.md`：人可读结果，并自动计算相对论文 51.8/59.2 的百分点差值。
- `cache/`：每个 API 请求的 prompt、图片哈希、原始响应、清洗响应、token 与延迟。

## 5. 统计口径

Task 2 是论文 Table 1 的主指标。对每条 gold merged entity，只有预测中的
`source_image_entities` 集合和 `source_text_entities` 集合都与 gold **完全相等**才计正确。
micro 是所有实体合并统计；macro 是先计算每篇文档准确率再等权平均。只有
`overall.total == 1114` 的报告可用于“超过论文”的正式结论；开发子集报告会自动标注
“不可作为正式超越结论”。缺失预测按错误处理。

论文 Task2 accuracy 实质上是对 gold links 的 recall：预测额外链接不扣分。本工作因此
额外报告唯一集合对的 micro precision/recall/F1。并集结果只作协议审计；二次裁决必须
同时提高论文 accuracy 和 F1 才会从 dev 扩到 full。

Task 1 衡量全图实体与文本实体的链接；Task 3 衡量 VLM 生成的全图实体名称。
报告会按当前方法输出适用任务，但“超越 Table 1”的判据只使用完整集 Task 2。

报告只展示当前方法实际生成的任务：embedding/LLM/clustering/ensemble 输出 Task1/2；
只有 `--method task3` 才输出 Task3。早期报告中 clustering 的 Task3=0 表示字段不存在，
不是模型能力结果；重新执行 `evaluate.py` 会将该不适用行移除。

## 6. 已修复的复现问题

1. SpecLink 对近似对称拉普拉斯矩阵使用 `eig` 会产生复数特征向量；改为显式对称化
   后使用 `eigh`。
2. DBSCAN+KNN 分支漏设 `target_label`；已补齐。
3. 原 LLM 标签解析失败时任意回退到簇 1；现在提取合法数值，失败则回退到语义最近邻。
4. JSON 正则无法可靠处理嵌套对象和 Markdown/thinking 包装；已改为 JSON decoder 扫描。
5. 旧缓存键未包含 endpoint、推理参数与 thinking 设置；现在使用 schema 2 隔离缓存，
   同时保存原始响应用于审计。
6. 标准 JSON decoder 仍可能拒绝模型生成的 `\'` 或未转义内部双引号；pipeline schema 3
   只在标准解析失败后使用 `json-repair`，并把 schema 写入每个新 run 配置。
7. 混合模型需要 text/mm 两套独立缓存命名空间；pipeline schema 4 已加入
   `CMEL_TEXT_CACHE_NAMESPACE` 和 `CMEL_MM_CACHE_NAMESPACE`。脚本使用“外部值优先”
   规则，避免再次把 `CMEL_MM_MODEL` 静默覆盖成文本模型。

实验性对齐路径通过 `CMEL_ALIGNMENT_PROMPT_VERSION` 选择并写入 run 配置。默认
`paper_v1` 保持上游行为；`strict_v2` 和 `hybrid_verify_v3` 已分别在 dev/mini 明显退化，
只作为有审计产物的负结果保留，不推荐运行。`CMEL_DENSE_CANDIDATE_TOP_K` 仅在后者使用，
默认 5。

上游仓库不被直接修改；实际实验始终使用 `vendor/cmel_research/` 中的源码。
补丁摘要保存在 `patches/`；vendor 文件本身是实验执行的唯一真源，避免补丁未应用时
误跑上游旧代码。
