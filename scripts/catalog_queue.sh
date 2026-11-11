#!/usr/bin/env bash
# The catalog's GPU stages in order (scripts/catalog_box.sh starts this), then what follows from their tables:
# the weight-only stages and the measuring half of the edges check (its `score` half runs where the
# comparison lenses are). A failure among the GPU stages stops the queue; after those, a failure is logged and
# the queue carries on. Rerun to resume: a finished stage leaves logs/<stage>.done and is skipped.
# MEDIA=1 also reruns Gemma 4's media exemplars (13, 14), which need about 220 GB of RAM.
# UPLOAD is the branch of the Hub dataset each table goes to as soon as its stage has run, in the background (empty for none).
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
UPLOAD="${UPLOAD-neuron-lens-2}"
up() {  # up <stage> <model> <tables>
    [ -n "$UPLOAD" ] && run "upload_$1" "$2" python scripts/12_release.py --tables "$3" --branch "$UPLOAD" &
}
Q=Qwen/Qwen3-8B
G=google/gemma-4-12B
must acts_qwen3-8b "$Q" python scripts/06_activations.py && up acts_qwen3-8b "$Q" acts,docs
must firings_qwen3-8b "$Q" python scripts/17_firings.py && up firings_qwen3-8b "$Q" firings,sources
must drivers_qwen3-8b "$Q" python scripts/07_drivers.py && up drivers_qwen3-8b "$Q" drivers
must acts_gemma "$G" python scripts/06_activations.py && up acts_gemma "$G" acts,docs
run firings_gemma "$G" python scripts/17_firings.py && up firings_gemma "$G" firings,sources
run drivers_gemma "$G" python scripts/07_drivers.py && up drivers_gemma "$G" drivers

run reads_qwen3-8b "$Q" python scripts/08_reads.py && up reads_qwen3-8b "$Q" reads
run reads_at_firings_qwen3-8b "$Q" python scripts/checks/reads_at_firings.py
run edges_qwen3-8b "$Q" python scripts/16_edges.py && up edges_qwen3-8b "$Q" edges
run reads_gemma "$G" python scripts/08_reads.py && up reads_gemma "$G" reads
run reads_at_firings_gemma "$G" python scripts/checks/reads_at_firings.py
run edges_gemma "$G" python scripts/16_edges.py && up edges_gemma "$G" edges
run edges_context_qwen3-8b "$Q" python scripts/checks/edges_in_context.py measure
run edges_context_gemma "$G" python scripts/checks/edges_in_context.py measure
if [ "${MEDIA:-0}" = 1 ]; then
    run media_image "$G" python scripts/13_media_activations.py --device cuda --modalities image
    run media_audio "$G" python scripts/13_media_activations.py --device cuda --modalities audio
    run media_crossmodal "$G" python scripts/checks/crossmodal.py
    run media_tables "$G" python scripts/14_media_tables.py && up media_tables "$G" image,audio,media
fi
wait
echo "$(date '+%F %T') QUEUE DONE"
