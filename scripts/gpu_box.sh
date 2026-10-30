#!/usr/bin/env bash
# First steps on a rented CUDA box, from a fresh checkout of this repo:
#
#   git clone https://github.com/stdlogits/neuron-lens && cd neuron-lens
#   DOCS=1000 bash scripts/gpu_box.sh
#   DOCS=1000 MODEL=google/gemma-4-12B bash scripts/gpu_box.sh   # Gemma 4 12B (gemma4_unified)
#
# Installs the package (no Drowse needed; uv picks the torch build that matches the CUDA
# driver), downloads the models, exports the lens corpus unless data/corpus/ came along, runs
# the estimator tests and benchmarks one document per lens. Then fit, R first, then S, on one
# GPU (about 3 hours per lens for 1,000 Qwen3-8B documents on an H100; Gemma 4 12B about 4.5):
#
#   python scripts/02_fit.py --lens R --model "$MODEL" --docs "$DOCS" --detach
#
# or one lens per GPU (CUDA_VISIBLE_DEVICES=0 ... --lens R, CUDA_VISIBLE_DEVICES=1 ... --lens S),
# and copy data/fits/ off the box before destroying it.
set -euo pipefail
cd "$(dirname "$0")/.."
DOCS="${DOCS:-1000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
# Rented boxes have up to ~200 cores; torch's CPU ops on small tensors then spend their time
# in thread spin-waits (the tests took minutes per case, and a CPU-side step stalled the GPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

command -v uv > /dev/null || python3 -m pip install --user --quiet uv
UV="$(command -v uv || echo "python3 -m uv")"
$UV venv --python 3.13 .venv
source .venv/bin/activate
$UV pip install --quiet --torch-backend auto -e . pytest
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); print('torch', torch.__version__, 'cuda', torch.version.cuda, (x @ x).float().norm().item() > 0)"

hf download Qwen/Qwen3-0.6B > /dev/null
hf download "$MODEL" > /dev/null
[ -f data/corpus/dclm.jsonl ] || python scripts/01_lens_corpus.py --docs "$DOCS"

python -m pytest -q tests/test_transport.py tests/test_gemma4.py
python scripts/checks/fit_bench.py --device cuda --models Qwen/Qwen3-0.6B,"$MODEL" \
    --dim-batch 128,256,512 --compare-fp32
