#!/usr/bin/env bash
# First steps on a rented CUDA box, from a fresh checkout of this repo:
#
#   git clone https://github.com/stdlogits/neuron-lens && cd neuron-lens
#   DOCS=1000 bash scripts/gpu_box.sh
#   DOCS=1000 MODEL=google/gemma-4-12B bash scripts/gpu_box.sh   # Gemma 4 12B (gemma4_unified)
#   DOCS=1000 MODEL=Qwen/Qwen3.5-35B-A3B SEQ_LEN=512 bash scripts/gpu_box.sh
#
# Qwen3.5 fits documents of exactly SEQ_LEN tokens (--full-length), so its DeltaNet memories
# fill: about 63% of DCLM rows reach 512 of its tokens and 33% reach 1,024, so the corpus is
# exported to CORPUS_ROWS rows (default 4,000 for Qwen3.5; the first 1,000 stay the same, so
# other models' fits are unchanged). Its decoder stack (68 GB in bf16) leaves ~15 GiB of an 80 GB H100:
# the sweep (perf_sweep.py) finds the dim batch, TF32, compile and sum-device settings.
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
SEQ_LEN="${SEQ_LEN:-128}"
case "$MODEL" in
    Qwen/Qwen3.5*|Qwen/Qwen3.6*) QWEN35=1; CORPUS_ROWS="${CORPUS_ROWS:-4000}" ;;
    *) QWEN35=0; CORPUS_ROWS="${CORPUS_ROWS:-$DOCS}" ;;
esac
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
rows=$( [ -f data/corpus/dclm.jsonl ] && wc -l < data/corpus/dclm.jsonl || echo 0 )
[ "$rows" -ge "$CORPUS_ROWS" ] || python scripts/01_lens_corpus.py --docs "$CORPUS_ROWS"

python -m pytest -q tests/test_transport.py tests/test_gemma4.py tests/test_qwen35.py
if [ "$QWEN35" = 1 ]; then
    # Precision on a 4-layer slice against fp32, then a first throughput grid on the whole model;
    # scripts/checks/perf_sweep.py's docstring has the rest of the sweep.
    python scripts/checks/perf_sweep.py --model "$MODEL" --seq-len "$SEQ_LEN" --fp32-layers 4 \
        --dim-batch 32 --tf32 0,1 --compile 0,1
    python scripts/checks/perf_sweep.py --model "$MODEL" --seq-len "$SEQ_LEN" \
        --dim-batch 16,32,64,128 --tf32 0,1 --compile 1
else
    python scripts/checks/fit_bench.py --device cuda --models Qwen/Qwen3-0.6B,"$MODEL" \
        --dim-batch 128,256,512 --compare-fp32
fi
