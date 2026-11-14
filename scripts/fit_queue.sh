#!/usr/bin/env bash
# The text lenses of a list of models on a machine with two GPUs, at the lenses of record's settings (the first DOCS
# documents of data/corpus/dclm.jsonl, 128 tokens): per model the scale, then the R-lens, on GPU 0, and the S-lens on
# GPU 1. Each model downloads when its turn comes and, past DROP_GB of weights, is deleted from the cache after its
# fits (the catalog stages download it again). Fits save every 10 minutes and resume when rerun; a finished one leaves
# logs/fit_<lens>_<model>.done and is skipped. A failed fit is logged, keeps its model, and the queue moves on.
# Progress in logs/fit_queue.log, each fit's output in logs/fit_<lens>_<model>.log. After scripts/gpu_box.sh:
#
#   setsid nohup bash scripts/fit_queue.sh Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B-Base ... > logs/fit_queue.log 2>&1 &
#
# A model may carry flags for its R and S fits after a colon, as one argument: "Qwen/Qwen3-32B:--offload 20".
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # large models fill the GPU
DOCS="${DOCS:-1000}"
DROP_GB="${DROP_GB:-10}"
mkdir -p logs
fit() {  # fit <gpu> <lens> <model> <stem> [02_fit.py flags]
    local gpu="$1" lens="$2" model="$3" name="fit_${2,,}_$4"; shift 4
    [ -e "logs/$name.done" ] && return 0
    echo "$(date '+%F %T') start $name"
    if CUDA_VISIBLE_DEVICES="$gpu" python scripts/02_fit.py --lens "$lens" --model "$model" --docs "$DOCS" \
            --device cuda "$@" >> "logs/$name.log" 2>&1; then
        touch "logs/$name.done"
        echo "$(date '+%F %T') done $name"
    else
        echo "$(date '+%F %T') FAILED $name: see logs/$name.log"
        return 1
    fi
}
failed=0
for spec in "$@"; do
    model="${spec%%:*}"
    flags=()
    [ "$spec" != "$model" ] && read -ra flags <<< "${spec#*:}"
    stem="$(tr '[:upper:]' '[:lower:]' <<< "${model##*/}")"
    echo "$(date '+%F %T') model $model ${flags[*]}"
    if ! hf download "$model" > /dev/null; then
        echo "$(date '+%F %T') FAILED download $model"
        failed=1
        continue
    fi
    { fit 0 scale "$model" "$stem" && fit 0 R "$model" "$stem" "${flags[@]}"; } &
    r=$!
    fit 1 S "$model" "$stem" "${flags[@]}" &
    s=$!
    wait $r; ok_r=$?
    wait $s; ok_s=$?
    if [ $ok_r != 0 ] || [ $ok_s != 0 ]; then
        failed=1
        continue
    fi
    gb=$(python - "$model" <<'EOF'
import sys
from huggingface_hub import scan_cache_dir
print(sum(r.size_on_disk for r in scan_cache_dir().repos if r.repo_type == "model" and r.repo_id == sys.argv[1]) >> 30)
EOF
)
    if [ "$gb" -gt "$DROP_GB" ] && hf cache rm -y "model/$model" > /dev/null; then
        echo "$(date '+%F %T') dropped $model from the cache"
    fi
done
echo "$(date '+%F %T') QUEUE DONE$([ $failed = 0 ] || echo ', with failures')"
exit $failed
