#!/bin/bash
set -euo pipefail
export TMPDIR=/data/zzhu126/tmp/deployment
export PIP_CACHE_DIR=/data/zzhu126/cache/pip
export XDG_CACHE_HOME=/data/zzhu126/cache
export HF_HOME=/data/zzhu126/cache/huggingface
export UV_CACHE_DIR=/data/zzhu126/cache/uv
export UV_PYTHON_INSTALL_DIR=/data/zzhu126/environments/uv-python
export TORCH_HOME=/data/zzhu126/cache/torch
export CUDA_CACHE_PATH=/data/zzhu126/cache/cuda
mkdir -p "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR"

export UV_PROJECT_ENVIRONMENT=/data/zzhu126/environments/ltx-2.5
cd /data/zzhu126/multi_object_benchmark/runtime/source_repos/LTX-2.5
/data/zzhu126/environments/ltx25-bootstrap/bin/uv sync --all-packages --extra natten --no-dev --python /usr/bin/python3.12 --no-cache
/data/zzhu126/environments/ltx-2.5/bin/python -m ltx_pipelines.distilled --help > /data/zzhu126/multi_object_benchmark/runtime/deployments/LTX-2.5/cli_help.txt
/data/zzhu126/environments/ltx25-bootstrap/bin/uv pip freeze --python /data/zzhu126/environments/ltx-2.5/bin/python > /data/zzhu126/multi_object_benchmark/runtime/deployments/LTX-2.5/environment.txt
touch /data/zzhu126/multi_object_benchmark/runtime/deployments/LTX-2.5/install.ok

