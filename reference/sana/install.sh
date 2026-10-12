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

source /data/zzhu126/environments/sana-video2/bin/activate
cd /data/zzhu126/multi_object_benchmark/runtime/source_repos/Sana-video2
python -m pip install -U pip wheel 'setuptools<80'
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 xformers==0.0.33.post2
pip install --no-cache-dir --no-build-isolation mmcv==1.7.2
pip install --no-cache-dir -e .
pip check
pip freeze > /data/zzhu126/multi_object_benchmark/runtime/deployments/Sana-video2/environment.txt
touch /data/zzhu126/multi_object_benchmark/runtime/deployments/Sana-video2/install.ok

