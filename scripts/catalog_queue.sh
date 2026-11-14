#!/usr/bin/env bash
# The catalog's stages for each model given (default Qwen/Qwen3-8B and google/gemma-4-12B; scripts/catalog_box.sh starts
# this), model by model: the GPU passes 06, 17 and 07, then what follows from their tables: 08, the reads-at-firings
# check, 16 and the measuring half of the edges check (its `score` half runs where the comparison lenses are). A failed
# GPU pass skips the rest of its model; after those, a failure is logged and the queue carries on. Rerun to resume: a
# finished stage leaves logs/<stage>_<model>.done and is skipped. Each model downloads at its turn; DROP=1 deletes it
# from the cache after its stages. With several GPUs, run one queue per GPU (CUDA_VISIBLE_DEVICES=i), each with its
# own models. MEDIA=1 also reruns Gemma 4's media stages (13, 14), which need about 220 GB of RAM. UPLOAD names the
# branch of the Hub dataset each table goes to as soon as its stage has run, in the background (unset: none).
#
#   UPLOAD=b setsid nohup bash scripts/catalog_queue.sh Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B >> logs/catalog_queue.log 2>&1 &
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
UPLOAD="${UPLOAD-}"
up() {  # up <stage> <model> <tables>
    [ -n "$UPLOAD" ] && run "upload_$1" "$2" python scripts/12_release.py --tables "$3" --branch "$UPLOAD" &
}
[ $# -gt 0 ] || set -- Qwen/Qwen3-8B google/gemma-4-12B
for m in "$@"; do
    s="$(tr '[:upper:]' '[:lower:]' <<< "${m##*/}")"
    echo "$(date '+%F %T') model $m"
    hf download "$m" > /dev/null || { echo "$(date '+%F %T') FAILED download $m"; continue; }
    run "acts_$s" "$m" python scripts/06_activations.py && up "acts_$s" "$m" acts,docs &&
        run "firings_$s" "$m" python scripts/17_firings.py && up "firings_$s" "$m" firings,sources &&
        run "drivers_$s" "$m" python scripts/07_drivers.py && up "drivers_$s" "$m" drivers || continue
    run "reads_$s" "$m" python scripts/08_reads.py && up "reads_$s" "$m" reads
    run "reads_at_firings_$s" "$m" python scripts/checks/reads_at_firings.py
    run "edges_$s" "$m" python scripts/16_edges.py && up "edges_$s" "$m" edges
    run "edges_context_$s" "$m" python scripts/checks/edges_in_context.py measure
    if [ "${MEDIA:-0}" = 1 ] && [ "$s" = gemma-4-12b ]; then
        run media_image "$m" python scripts/13_media_activations.py --device cuda --modalities image
        run media_audio "$m" python scripts/13_media_activations.py --device cuda --modalities audio
        run media_crossmodal "$m" python scripts/checks/crossmodal.py
        run media_tables "$m" python scripts/14_media_tables.py && up media_tables "$m" image,audio,media
    fi
    if [ "${DROP:-0}" = 1 ] && hf cache rm -y "model/$m" > /dev/null; then
        echo "$(date '+%F %T') dropped $m from the cache"
    fi
done
wait
echo "$(date '+%F %T') QUEUE DONE"
