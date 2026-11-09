#!/usr/bin/env bash
# The catalog's GPU stages in order (scripts/catalog_box.sh starts this), then what follows from their tables:
# the weight-only stages and the measuring halves of the causal checks (their `score` halves run where the
# comparison lenses are). A failure among the GPU stages stops the queue; after those, a failure is logged and
# the queue carries on. Rerun to resume: a finished stage leaves logs/<stage>.done and is skipped.
# MEDIA=1 also reruns Gemma 4's media exemplars (13, 14), which need about 220 GB of RAM.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
run() {
    local name="$1" model="$2"; shift 2
    [ -e "logs/$name.done" ] && return 0
    echo "$(date '+%F %T') start $name"
    if NEURON_LENS_MODEL="$model" "$@" >> "logs/$name.log" 2>&1; then
        touch "logs/$name.done"
        echo "$(date '+%F %T') done $name"
    else
        echo "$(date '+%F %T') FAILED $name: see logs/$name.log"
        return 1
    fi
}
must() { run "$@" || exit 1; }
Q=Qwen/Qwen3-8B
G=google/gemma-4-12B
must acts_qwen3-8b "$Q" python scripts/06_activations.py
must firings_qwen3-8b "$Q" python scripts/17_firings.py
must sources_qwen3-8b "$Q" python scripts/18_sources.py
must acts_gemma "$G" python scripts/06_activations.py
must firings_gemma "$G" python scripts/17_firings.py

run reads_qwen3-8b "$Q" python scripts/08_reads.py --parts neurons
run legible_qwen3-8b "$Q" python scripts/09_legible.py
run edges_qwen3-8b "$Q" python scripts/16_edges.py
run reads_gemma "$G" python scripts/08_reads.py --parts neurons
run legible_gemma "$G" python scripts/09_legible.py
run causal_qwen3-8b "$Q" python scripts/checks/causal_lens.py measure
run causal_gemma "$G" python scripts/checks/causal_lens.py measure --per-layer 24
run context_qwen3-8b "$Q" python scripts/checks/context_linearity.py measure
run edges_context_qwen3-8b "$Q" python scripts/checks/edges_in_context.py measure
if [ "${MEDIA:-0}" = 1 ]; then
    run media_image "$G" python scripts/13_media_activations.py --device cuda --modalities image
    run media_audio "$G" python scripts/13_media_activations.py --device cuda --modalities audio
    run media_crossmodal "$G" python scripts/checks/crossmodal.py
    run media_tables "$G" python scripts/14_media_tables.py
fi
echo "$(date '+%F %T') QUEUE DONE"
