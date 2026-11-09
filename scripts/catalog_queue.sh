#!/usr/bin/env bash
# The catalog's GPU stages in order (scripts/catalog_box.sh starts this). Stops at the first failure;
# rerun to resume. MEDIA=1 also reruns Gemma 4's media exemplars (13, 14), which need about 220 GB of RAM.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
run() {
    local name="$1" model="$2"; shift 2
    echo "$(date '+%F %T') start $name"
    if ! NEURON_LENS_MODEL="$model" "$@" >> "logs/$name.log" 2>&1; then
        echo "$(date '+%F %T') FAILED $name: see logs/$name.log"; exit 1
    fi
    echo "$(date '+%F %T') done $name"
}
Q=Qwen/Qwen3-8B
G=google/gemma-4-12B
run acts_qwen3-8b "$Q" python scripts/06_activations.py
run firings_qwen3-8b "$Q" python scripts/17_firings.py
run sources_qwen3-8b "$Q" python scripts/18_sources.py
run acts_gemma "$G" python scripts/06_activations.py
run firings_gemma "$G" python scripts/17_firings.py
if [ "${MEDIA:-0}" = 1 ]; then
    run media_image "$G" python scripts/13_media_activations.py --device cuda --modalities image
    run media_audio "$G" python scripts/13_media_activations.py --device cuda --modalities audio
    run media_crossmodal "$G" python scripts/checks/crossmodal.py
    run media_tables "$G" python scripts/14_media_tables.py
fi
echo "$(date '+%F %T') QUEUE DONE"
