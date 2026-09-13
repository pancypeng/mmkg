#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:-}"
if [[ -z "${MODEL_KEY}" ]]; then
  echo "usage: $0 {27b|flash} [extra vLLM args...]" >&2
  exit 2
fi
shift

CMEL_PORT="${CMEL_PORT:-8000}"
MODEL_ARGS=()

case "${MODEL_KEY}" in
  27b)
    MODEL_PATH=/root/models/Qwen3.8-27B
    SERVED_NAME=qwen38-27b
    GPU_UTILIZATION="${GPU_UTILIZATION:-0.75}"
    VLLM_BIN="${VLLM_BIN:-/mnt/conda-store/envs/qwen-vllm/bin/vllm}"
    ;;
  flash)
    MODEL_PATH=/root/models/Qwen3.8-Flash-Next-FP8
    SERVED_NAME=qwen38-flash-next-fp8
    # L20X 上编译期会为 Ngram 测试张量额外申请约 23.84 GiB，导致启动 OOM。
    # eager 模式已在 2 x L20X 上验证；仍保留约 20 GiB/卡供 Stella 和运行时使用。
    GPU_UTILIZATION="${GPU_UTILIZATION:-0.85}"
    VLLM_BIN="${VLLM_BIN:-/mnt/conda-store/envs/qwen-flash-vllm/bin/vllm}"
    MODEL_ARGS=(--no-enable-flashinfer-autotune --moe-backend triton)
    FLASH_EXECUTION_MODE="${FLASH_EXECUTION_MODE:-decode_graph}"
    case "${FLASH_EXECUTION_MODE}" in
      eager)
        # 最保守的兼容回退；吞吐较低，但可完全禁用 compile/CUDA graph。
        MODEL_ARGS+=(--enforce-eager)
        ;;
      decode_graph)
        # 官方低显存 recipe：不做 inductor compile，仅为 decode 捕获 CUDA graph。
        MODEL_ARGS+=(-cc.mode=none -cc.cudagraph_mode=full_decode_only)
        ;;
      *)
        echo "unknown FLASH_EXECUTION_MODE: ${FLASH_EXECUTION_MODE}; expected eager or decode_graph" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "unknown model key: ${MODEL_KEY}; expected 27b or flash" >&2
    exit 2
    ;;
esac

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${CMEL_PORT}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_UTILIZATION}" \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --enable-prefix-caching \
  "${MODEL_ARGS[@]}" \
  "$@"
