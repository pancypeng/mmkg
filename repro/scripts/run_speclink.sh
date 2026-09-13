#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-}"
RUN_NAME="${2:-}"
SPLIT="${3:-dev}"
if [[ -z "${MODEL_NAME}" || -z "${RUN_NAME}" ]]; then
  echo "usage: $0 MODEL_NAME RUN_NAME {mini|dev|full}" >&2
  exit 2
fi

case "${SPLIT}" in
  mini) DATA_ARGS=(--docs 1 --seed 0) ;;
  dev) DATA_ARGS=(--docs 4 --seed 0) ;;
  full) DATA_ARGS=(--all-docs --seed 0) ;;
  *) echo "unknown split: ${SPLIT}; expected mini, dev or full" >&2; exit 2 ;;
esac

export CMEL_EMBED_MODEL="${CMEL_EMBED_MODEL:-/root/models/stella_en_1.5B_v5}"
export CMEL_EMBED_DEVICE="${CMEL_EMBED_DEVICE:-cuda:0}"
export CMEL_API_KEY="${CMEL_API_KEY:-local}"
export CMEL_API_BASE="${CMEL_API_BASE:-http://127.0.0.1:8000/v1}"
export CMEL_MODEL="${CMEL_MODEL:-${MODEL_NAME}}"
export CMEL_MM_API_KEY="${CMEL_MM_API_KEY:-${CMEL_API_KEY}}"
export CMEL_MM_API_BASE="${CMEL_MM_API_BASE:-${CMEL_API_BASE}}"
export CMEL_MM_MODEL="${CMEL_MM_MODEL:-${MODEL_NAME}}"
export CMEL_LLM_TEMPERATURE="${CMEL_LLM_TEMPERATURE:-0}"
export CMEL_LLM_TOP_P="${CMEL_LLM_TOP_P:-1}"
export CMEL_LLM_TOP_K="${CMEL_LLM_TOP_K:--1}"
export CMEL_LLM_PRESENCE_PENALTY="${CMEL_LLM_PRESENCE_PENALTY:-0}"
export CMEL_LLM_MAX_TOKENS="${CMEL_LLM_MAX_TOKENS:-2048}"
export CMEL_LLM_SEED="${CMEL_LLM_SEED:-0}"
export CMEL_ENABLE_THINKING="${CMEL_ENABLE_THINKING:-0}"
export CMEL_CACHE_NAMESPACE="${CMEL_CACHE_NAMESPACE:-${MODEL_NAME}-nonthink-t0-v2}"
export CMEL_TEXT_CACHE_NAMESPACE="${CMEL_TEXT_CACHE_NAMESPACE:-${CMEL_CACHE_NAMESPACE}}"
export CMEL_MM_CACHE_NAMESPACE="${CMEL_MM_CACHE_NAMESPACE:-${CMEL_CACHE_NAMESPACE}}"
export CMEL_ALIGNMENT_PROMPT_VERSION="${CMEL_ALIGNMENT_PROMPT_VERSION:-paper_v1}"
export CMEL_DENSE_CANDIDATE_TOP_K="${CMEL_DENSE_CANDIDATE_TOP_K:-5}"

PYTHON_BIN="${CMEL_PYTHON:-/private/mmkg/.venv-stella310/bin/python}"
exec "${PYTHON_BIN}" -u run_cmel.py \
  --method clustering --clustering spectral --classify llm \
  "${DATA_ARGS[@]}" --run "${RUN_NAME}" --mode api
