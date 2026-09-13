# MMGraphRAG 实验状态

最后更新：2026-09-11T04:59:55Z（UTC）

本文件是跨会话恢复入口；稳定操作说明见 `README.md`，每次实验的完整产物见
`runs/<run_name>/`。

## 当前目标与正式判据

- 在 CMEL 完整 Task 2（1,114 条 gold）上比较 Qwen3.8-27B 与
  Qwen3.8-Flash-Next-FP8。
- 论文 Spe-L 目标：Overall 51.8% micro / 59.2% macro。
- 只有 `metrics.json` 中 `task2.overall.total == 1114` 才能声称完整集超过论文；
  12 篇 dev 结果只用于选配置。

## 已完成

### 数据和环境

- 数据预检通过：77 篇带 Task 2 标注的文档、678 张图片、1,114 条标注、缺图 0。
- 统一模型目录：`/root/models`；清单为 `/root/models/models.json`。
- Qwen3.8-27B 已在双 L20X、TP=2 下完成文本与真实图片 smoke test。
- Qwen3.8-Flash-Next-FP8 权重已完整下载；vLLM 0.29.0 能识别架构并加载权重。
- 主程序已支持单图 checkpoint、逐文档指标刷新、隔离缓存、token/延迟统计、
  配置冲突保护与完整数据集运行。

### Qwen3.8-27B 固定 dev12 基线

Run：`runs/qwen38_27b_spec_llm_dev12_t0/`

| 指标 | 结果 |
|---|---:|
| 图片 | 136 / 136，失败 0 |
| Task 1 overall | 55.9% micro / 50.4% macro |
| Task 2 news | 50.0% micro / 52.1% macro，n=16 |
| Task 2 novel | 43.1% micro / 45.3% macro，n=116 |
| Task 2 paper | 71.9% micro / 76.5% macro，n=64 |
| Task 2 overall | **53.1% micro / 58.0% macro，n=196** |
| 相对论文完整集 Spe-L | +1.3 pp micro / -1.2 pp macro（仅 dev，不作正式结论） |
| 调用 | 922 text + 136 multimodal |
| tokens | 3,604,977 prompt + 57,300 completion = 3,662,277 |
| API 累计延迟 | 757.047 s |
| 端到端耗时 | 813 s |

生成配置：temperature=0、top_p=1、top_k=-1、presence_penalty=0、seed=0、
max_tokens=2048、关闭 thinking；EMB 为 Stella-en-1.5B-v5。

### Qwen3.8-Flash-Next-FP8 固定 dev12

Run：`runs/qwen38_flash_spec_llm_dev12_t0/`

| 指标 | 结果 |
|---|---:|
| 图片 | 136 / 136，失败 0 |
| Task 1 overall | 59.6% micro / 57.8% macro |
| Task 2 news | 62.5% micro / 62.5% macro，n=16 |
| Task 2 novel | 46.6% micro / 53.6% macro，n=116 |
| Task 2 paper | 71.9% micro / 73.3% macro，n=64 |
| Task 2 overall | **56.1% micro / 63.1% macro，n=196** |
| Flash − 27B | **+3.0 pp micro / +5.1 pp macro** |
| 相对论文完整集 Spe-L | +4.3 pp micro / +3.9 pp macro（仅 dev，不作正式结论） |
| 累计逻辑调用 | 909 text + 135 multimodal；57 cache hits |
| 已记录 API tokens | 3,546,366 prompt + 63,353 completion = 3,609,719 |
| 已记录累计耗时 | 1,080.048 s（包含 eager 首篇与一次中断/恢复） |

生成配置和 27B 完全一致。运行曾在首篇后有控制式中断，结果无损；因此这里的调用延迟
用于恢复审计，不用于严谨吞吐对比。逐例输出、机器指标与独立报告均已落盘。

### Qwen3.8-Flash-Next-FP8 完整集 temperature=0 基线

Run：`runs/qwen38_flash_spec_llm_full_t0/`

| 指标 | 结果 |
|---|---:|
| 图片/失败 | 678 / 0 |
| Task 1 overall | 55.5% micro / 53.6% macro，n=678 |
| Task 2 news | 58.6% micro / 54.7% macro，n=87 |
| Task 2 novel | 32.6% micro / 40.5% macro，n=552 |
| Task 2 paper | 64.2% micro / 62.8% macro，n=475 |
| Task 2 overall | **48.1% micro / 55.4% macro，n=1,114** |
| 相对论文 Spe-L | **-3.7 pp micro / -3.8 pp macro** |
| 实际 API / 缓存命中 | 4,639 / 1,342 |
| 已记录 tokens | 16,430,560 prompt + 293,804 completion = 16,724,364 |
| 端到端耗时 | 3,986.165 s（66.4 min） |

完整性硬检查全部通过，但该配置没有超过论文。开发子集高估了 paper 域泛化：dev
71.9/73.3，完整集只有 64.2/62.8。基线保留不覆盖。

### Qwen3.8-27B 完整集 temperature=0 基线

Run：`runs/qwen38_27b_spec_llm_full_t0_schema4/`

| 指标 | 结果 |
|---|---:|
| 图片/失败 | 678 / 0 |
| Task 1 overall | 54.7% micro / 52.9% macro，n=678 |
| Task 2 news | 47.1% micro / 45.4% macro，n=87 |
| Task 2 novel | 33.3% micro / 34.1% macro，n=552 |
| Task 2 paper | 67.4% micro / 66.3% macro，n=475 |
| Task 2 overall | **48.9% micro / 51.8% macro，n=1,114** |
| 相对论文 Spe-L | **-2.9 pp micro / -7.4 pp macro** |
| API / 缓存命中 | 4,781 / 1,200；错误 0 |
| tokens | 16,877,656 prompt + 290,543 completion = 17,168,199 |
| 端到端耗时 | 4,033 s（67.2 min） |

单模型同样未超过论文。27B 的 paper micro 略高于 Flash，但 news 明显更低，完整集两模型
互补：27B 独对 122 条、Flash 独对 113 条、共同对 423 条。

### Flash + 27B 候选集成：完整集正式结果

协议审计 run：`runs/qwen38_flash_27b_union_full_protocol_audit/`。直接并集为
59.1%/63.6%，但 2,151 条预测中只有 658 条正确，precision=30.6%、F1=40.3%，不把它
单独当作真实提升。

冻结裁决 run：`runs/qwen38_flash_27b_adjudicated_by_27b_full_v1/`。

| 指标 | 结果 |
|---|---:|
| 图片/文档/Task2 gold/缺失 | 678 / 77 / 1,114 / 0 |
| Task 2 news | 58.6% micro / 54.7% macro |
| Task 2 novel | 40.8% micro / 51.0% macro |
| Task 2 paper | 73.9% micro / 72.7% macro |
| Task 2 overall | **56.3% micro / 61.3% macro** |
| 相对论文 Spe-L | **+4.5 pp micro / +2.1 pp macro** |
| 链接 precision / recall / F1 | 33.1% / 56.3% / 41.7% |
| 裁决调用 | 568 text，错误 0；2,231,598 tokens |
| 裁决端到端耗时 | 255.373 s |

这是在 dev12 冻结提示词与 temperature=0 参数后执行的无 gold 输入方法；27B 只能返回
已有候选 ID，不能创建、拆分或组合实体集合。它在论文完整口径上正式超过 Spe-L 两项
overall 指标，同时 F1 高于 Flash 单模的 40.3%，但低于 27B 单模的 43.3%；因此报告同时
保留 precision/F1，避免掩盖论文 accuracy 不惩罚假阳性的局限。

## Flash 启动问题与已验证修复

已排除的兼容性问题：vLLM 0.28.0 无法识别
`Qwen4ExpForConditionalGeneration`；独立环境升级到 vLLM 0.29.0 后架构识别和
131 个 shard 加载均成功（每卡约 86.92 GiB）。

首次失败发生在模型加载后的 torch.compile/profile 阶段，而非权重加载：进程每卡已用
约 137.56 GiB、仅余约 2.23 GiB 时，Ngram 测试张量继续申请 23.84 GiB 导致 OOM。
保守修复参数为 `GPU_UTILIZATION=0.85` 与 `--enforce-eager`。Flash 随后成功启动：每卡权重
86.92 GiB、峰值 activation 2.08 GiB、KV cache 28.19 GiB，32,768-token 请求理论并发
49.27 倍；无需 CPU offload。文本 smoke test 精确返回 `{"ok": true}`，真实 CMEL
图片正确描述了担架上的受伤儿童和伤员；完整数据/API 预检通过。随后验证官方
`mode=none + cudagraph_mode=full_decode_only` 同样稳定，并将新鲜文档吞吐从 eager 的
约 46 秒/图提高到约 5.3 秒/图；现已作为脚本默认值，eager 保留为回退。

## 已完成的后续消融

| 配置 | 范围 | Task 2 micro/macro | 结论 |
|---|---:|---:|---|
| Flash 官方 non-thinking 采样（seed=1） | dev12, n=196 | 22.4% / 29.4% | 显著退化，淘汰 |
| Flash thinking-low | mini, 首篇后中止 | 50.0% / 50.0%（仅首篇） | 慢约 27 s/图且首篇无增益，不完整 run |
| Flash t0 + schema-3 JSON repair | full, n=1,114 | 48.1% / 55.4% | 与原 full 完全相同；修复只提高鲁棒性 |
| Flash-VLM + 27B-LLM | dev12, n=196 | 53.1% / 58.0% | Task 2 与纯 27B 相同，低于纯 Flash |
| strict_v2 最小集合提示 | dev12, n=196 | 42.3% / 45.9% | paper 暴跌至 28.1/22.3，淘汰 |
| hybrid_verify_v3, dense K=5 | mini, n=87 | 19.5% / 8.2% | news/paper 为 0，淘汰 |
| Flash ∪ 27B（协议审计） | dev12, n=196 | 67.9% / 72.5% | precision 33.2%，不可单独作真实提升结论 |
| Flash+27B 候选经 27B 二次裁决 | dev12, n=196 | **66.3% / 70.0%** | 无 gold 输入；通过 full 门槛 |

有效混合 run 为 `runs/qwen38_hybrid_flashvlm_27bllm_dev12_t0_valid/`。其
`run_config.json` 已核对为 `text_model=qwen38-27b`、
`mm_model=qwen38-flash-next-fp8`，并使用独立 text/mm cache namespace。Task 1 提升到
58.8%/56.4%，但 Task 2 未提升，说明当前细粒度融合更受文本判定链路影响。早先同名
无 `valid` 后缀的 run 因脚本覆盖模型变量而无效，已由 `INVALID_RUN.md` 隔离。

误差分析脚本 `analyze_task2_errors.py` 已落盘。Flash dev12 的 86 个错误中，有 64 个
属于“图像实体集合正确、文本集合错误”；Flash 与 27B 任一答对即算对的严格 oracle 为
133/196=67.9%（仅诊断，不是可报告分数）。完整集的 578 个错误中相应类别为 320 个，
完全无重叠为 202 个。逐例报告保存在各 run 的 `task2_error_analysis.{json,md}`。

可执行二次裁决 run 为 `runs/qwen38_flash_27b_adjudicated_by_27b_dev12_v1/`。它只允许
27B 从两条基线已经产生的精确候选集合中选择，不允许新造、拆分或组合集合；证据包含
图像实体描述、文本实体描述与邻近文本。相对 Flash 基线，论文 Task2 accuracy 提高
10.2/6.9 pp；假阳性敏感的链接 F1 也从 46.6% 提到 48.1%。因此满足预设的 full 门槛。

## 下一步（按顺序）

1. 当前主目标已达成；保留所有服务、缓存、源码和 run，不覆盖正式结果。
2. 若继续提高可信度，按相同冻结配置增加 seed=1/2，并报告均值、方差和每次单独结果。
3. 若继续提高假阳性敏感质量，在新的开发划分上优化裁决阈值，以 27B 单模 F1=43.3%
   为门槛；不可再根据当前完整集 gold 修改 v1 后宣称同一次测试集泛化。

## 最终验证

- 核心源码、分析/集成脚本和 vendor 修复全部通过 `py_compile`。
- oracle/null 自检对 Task1/2/3 分别为 100%/0%，全部 PASS。
- preflight：两套 Qwen 与 Stella 路径存在；2×L20X 可见；77 篇、678 图、1,114 条
  Task2、缺图 0；本地 `qwen38-27b` API 健康。
- 四个完整 run 均独立核对磁盘 `result.json`：678 图；正式裁决 `progress.json` 为
  `complete`、678/678，`metrics.json` 为 1,114 条、77 篇、missing=0。
- 当前 27B vLLM 服务保持运行，便于下一会话继续；未删除任何模型、缓存或 run。

## 恢复前检查

```bash
cd /private/mmkg/repro
nvidia-smi
/private/mmkg/.venv-stella310/bin/python -m py_compile \
  config.py data.py llm_backend.py run_cmel.py evaluate.py
/private/mmkg/.venv-stella310/bin/python scripts/preflight.py --skip-api
```

不要删除 `cache/` 或已有 `runs/`。续跑必须复用相同 run 名、模型名、生成参数和
`CMEL_CACHE_NAMESPACE`；配置不一致时程序会主动终止。
