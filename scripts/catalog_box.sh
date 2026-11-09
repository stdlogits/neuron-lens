#!/usr/bin/env bash
# The catalog passes that need a GPU, on a rented box (GH200, or any CUDA machine with about 96 GB of
# GPU memory and 400 GB of RAM):
#
#   git clone -b neuron-lens-2 https://github.com/stdlogits/neuron-lens && cd neuron-lens
#   export HF_TOKEN=hf_...          # Gemma 4 is gated; the released lenses are read from the Hub
#   bash scripts/catalog_box.sh
#
# Installs the package and torch (a GH200 is aarch64: uv picks the CUDA build there too; set TORCH_ARGS
# if it cannot, for example TORCH_ARGS="--index-url https://download.pytorch.org/whl/cu128"), downloads
# the models, rebuilds the corpora at the revisions in data/corpus/*.provenance.json and the token
# counts, and starts scripts/catalog_queue.sh detached. Progress is in logs/catalog_queue.log, each
# stage's output in logs/<stage>.log. Every stage resumes when the queue is rerun.
#
# When the queue is done, copy the results to the machine that keeps the catalog and destroy the box:
#
#   rsync -av --include='*/' --include='*.parquet' --exclude='*' box:neuron-lens/data/catalog/ data/catalog/
#   rsync -av box:neuron-lens/data/acts/ data/acts/
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_TOKEN:?export HF_TOKEN first (Gemma 4 is gated)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_ENABLE_HF_TRANSFER=0

command -v uv > /dev/null || python3 -m pip install --user --quiet uv
UV="$(command -v uv || echo "python3 -m uv")"
[ -d .venv ] || $UV venv --python 3.13 .venv
source .venv/bin/activate
$UV pip install --quiet ${TORCH_ARGS:---torch-backend auto} -e '.[media]' pytest
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_properties(0).total_memory >> 30, 'GiB')"
python -m pytest -q tests/test_neurons.py tests/test_firings.py tests/test_sources.py

for model in Qwen/Qwen3-8B google/gemma-4-12B; do hf download "$model" > /dev/null; done
mkdir -p logs
if [ ! -s data/corpus/act-dclm.jsonl ]; then
    python scripts/05_act_corpus.py --docs dclm=40000,zh=8000,code=8000,chat=8000
    NEURON_LENS_MODEL=google/gemma-4-12B python scripts/05_act_corpus.py --docs chat=8000
fi
[ -s data/token_counts.pt ] || python scripts/03_token_counts.py
[ -s data/token_counts-gemma-4-12b.pt ] || NEURON_LENS_MODEL=google/gemma-4-12B python scripts/03_token_counts.py

setsid nohup bash scripts/catalog_queue.sh >> logs/catalog_queue.log 2>&1 < /dev/null &
echo "queue started: tail -f logs/catalog_queue.log"
