#!/usr/bin/env bash
# The catalog passes that need a GPU, on a CUDA machine with about 96 GB of GPU memory, 300 GB of RAM
# and 420 GB of disk. MEDIA=1 in the environment adds Gemma 4's media stages:
#
#   git clone -b neuron-lens-2 https://github.com/stdlogits/neuron-lens && cd neuron-lens
#   hf auth login                   # or export HF_TOKEN: Gemma 4 is gated
#   bash scripts/catalog_box.sh
#
# Installs the package and torch (uv picks the CUDA build; set TORCH_ARGS if it cannot, for example
# TORCH_ARGS="--index-url https://download.pytorch.org/whl/cu128"), downloads the models, builds any
# corpus or token count not already in data/ (at the revisions in data/corpus/*.provenance.json), and starts
# scripts/catalog_queue.sh detached; SETUP_ONLY=1 stops before the queue. Progress is in
# logs/catalog_queue.log, each stage's output in logs/<stage>.log. A rerun of the queue skips finished
# stages and resumes the long ones from their checkpoints.
#
# When the queue is done, copy the results to the machine that keeps the catalog and destroy the box:
#
#   rsync -av --include='*/' --include='*.parquet' --include='*.pt' --exclude='*' box:neuron-lens/data/catalog/ data/catalog/
#   rsync -av box:neuron-lens/data/acts/ data/acts/
#   rsync -av box:neuron-lens/data/causal/ data/causal/
#   rsync -av box:neuron-lens/data/summary/ data/summary/
#   rsync -av box:neuron-lens/logs/ logs/box/
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_ENABLE_HF_TRANSFER=0

if [ "${MEDIA:-0}" = 1 ]; then  # the synthetic images are drawn in these
    $(command -v sudo) apt-get update -qq || true
    $(command -v sudo) apt-get install -y -qq fonts-noto-core fonts-noto-cjk fonts-noto-extra
fi

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
UV=uv
[ -d .venv ] || $UV venv --python 3.13 .venv
source .venv/bin/activate
$UV pip install --quiet ${TORCH_ARGS:---torch-backend auto} -e '.[media]'
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_properties(0).total_memory >> 30, 'GiB')"
hf auth whoami > /dev/null || { echo "log in first (hf auth login, or export HF_TOKEN): Gemma 4 is gated"; exit 1; }

for model in Qwen/Qwen3-8B google/gemma-4-12B; do hf download "$model" > /dev/null; done
mkdir -p logs
build() {  # a slice's file, unless it is already complete
    local model="$1" name="$2" n="$3" file="$4"
    [ -s "$file" ] && [ "$(wc -l < "$file")" -ge "$n" ] || NEURON_LENS_MODEL="$model" python scripts/05_act_corpus.py --docs "$name=$n"
}
build Qwen/Qwen3-8B dclm 40000 data/corpus/act-dclm.jsonl
build Qwen/Qwen3-8B zh 8000 data/corpus/act-zh.jsonl
build Qwen/Qwen3-8B code 8000 data/corpus/act-code.jsonl
build Qwen/Qwen3-8B chat 8000 data/corpus/act-chat-qwen3-8b.jsonl
build google/gemma-4-12B chat 8000 data/corpus/act-chat-gemma-4-12b.jsonl
[ -s data/token_counts-qwen3-8b.pt ] || python scripts/03_token_counts.py
[ -s data/token_counts-gemma-4-12b.pt ] || NEURON_LENS_MODEL=google/gemma-4-12B python scripts/03_token_counts.py

[ "${SETUP_ONLY:-0}" = 1 ] && { echo "setup done"; exit 0; }
setsid nohup bash scripts/catalog_queue.sh >> logs/catalog_queue.log 2>&1 < /dev/null &
echo "queue started: tail -f logs/catalog_queue.log"
