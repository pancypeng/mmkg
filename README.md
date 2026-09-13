# MMKG / MMGraphRAG reproduction

This repository contains the MMGraphRAG source snapshot and a reproducible CMEL
Task 2 evaluation pipeline. The current frozen ensemble
(`Qwen3.8-Flash/27B candidates -> Qwen3.8-27B adjudication`) reaches **56.3%
micro / 61.3% macro accuracy** on the full 77-document, 1,114-gold-pair split.

## Repository layout

- `MMGraphRAG-main/`: upstream MMGraphRAG implementation and reference examples.
- `repro/`: reproducible experiment runner, evaluator, model-serving scripts,
  patches, environment notes, and curated Markdown results.
- `repro/README.md`: complete setup, preflight, serving, execution, resume, and
  result-statistics instructions.
- `repro/EXPERIMENT_STATUS.md`: experiment ledger and verified aggregate scores.
- `repro/MMGraphRAG_STEP_BY_STEP_REPORT.md`: step-by-step examples, intermediate
  outputs, model comparisons, and limitations.

## Data and model weights

Datasets, model weights, virtual environments, inference caches, and raw run
artifacts are intentionally excluded by `.gitignore`. They must be prepared
locally before running experiments. The expected local paths and model launch
parameters are documented in `repro/README.md`.

In particular, do not commit CMEL data or files such as `*.safetensors`,
`*.ckpt`, `*.pt`, `*.bin`, or `*.gguf` to this repository.

## Python dependencies

The root [`requirements.txt`](requirements.txt) covers the MMGraphRAG core and
the CMEL reproduction/evaluation pipeline. Python 3.10 is the validated version.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU use, install the PyTorch wheel matching the local CUDA driver before the
last command. MinerU and the two vLLM model-serving environments are intentionally
installed separately; their exact versions and launch commands are documented in
[`repro/README.md`](repro/README.md).

## Quick start

```bash
cd repro
./scripts/serve_qwen.sh 27b
/private/mmkg/.venv-stella310/bin/python scripts/preflight.py
./scripts/run_speclink.sh qwen38-27b qwen38_27b_spec_llm_dev12_t0 dev
```

See [`repro/README.md`](repro/README.md) for the full environment and experiment
procedure.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
