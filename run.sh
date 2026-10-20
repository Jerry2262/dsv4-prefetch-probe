#!/usr/bin/env bash
# Run the DSV4 prefetch probe in the existing 4090 environment.
#
# Two steps on the 4090 machine:
#   1) source /workspace/deepseek-v4-sm89/.venv/bin/activate
#   2) ./run.sh
#
# With no arguments this runs 8K / 32K / 64K context at batch 8.
# Any extra arguments are passed to bench_stages.py, for example:
#   ./run.sh --contexts 8192 --batches 1,8,16
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-/workspace/deepseek-v4-sm89}"

if [[ ! -d "${SGLANG_ROOT}/python/sglang" ]]; then
  echo "Expected the deepseek-v4-sm89 checkout at ${SGLANG_ROOT}" >&2
  echo "If it is elsewhere, run: SGLANG_ROOT=/path/to/deepseek-v4-sm89 ./run.sh" >&2
  exit 2
fi

# Prefer the known working venv; fall back to the activated python.
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${SGLANG_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${SGLANG_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  echo "Activate the environment first:" >&2
  echo "  source ${SGLANG_ROOT}/.venv/bin/activate" >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c "import torch, triton" >/dev/null 2>&1; then
  echo "This Python cannot import torch/triton: ${PYTHON_BIN}" >&2
  echo "Activate the working environment first:" >&2
  echo "  source ${SGLANG_ROOT}/.venv/bin/activate" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  set -- --contexts 8192,32768,65536 --batches 8 --warmup 10 --iters 50
fi

export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

echo "Python : ${PYTHON_BIN}"
echo "SGLang : ${SGLANG_ROOT}"
echo "Probe  : ${REPO_ROOT}/bench_stages.py"
echo "Args   : $*"
echo

exec "${PYTHON_BIN}" bench_stages.py "$@"
