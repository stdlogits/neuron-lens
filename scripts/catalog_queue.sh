#!/usr/bin/env bash
# The catalog's stages for each model given (default Qwen/Qwen3-8B and google/gemma-4-12B; scripts/catalog_box.sh starts
# this), model by model: the GPU passes 06, 17 and 07, then what follows from their tables: 08, the reads-at-firings
# check, 16, the edges check (checks/edges_in_context.py: measure, then score) and the
# steering check of the released R-lens (checks/causal_lens.py: with reads-at-firings, the site's depth bands). A failed
# GPU pass skips the rest of its model; after those, a failure is logged and the queue carries on. Rerun to resume: a
# finished stage leaves logs/<stage>_<model>.done and is skipped. Each model downloads at its turn; DROP=1 deletes it
# from the cache after its stages. With several GPUs, run one queue per GPU (CUDA_VISIBLE_DEVICES=i), each with its
# own models. MEDIA=1 also reruns Gemma 4's media stages (13, 14), which need about 220 GB of RAM. UPLOAD names the
# branch of the Hub dataset each table goes to as soon as its stage has run, in the background (unset: none); the plain
# files (token counts, the causal sample) go after the steering check.
# FIRINGS_SAVE_MINUTES=0 keeps 17's state in memory only, where its checkpoints don't fit the disk (Qwen3-14B: 91 GB).
#
#   UPLOAD=b setsid nohup bash scripts/catalog_queue.sh Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B >> logs/catalog_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # large models fill the GPU
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
VRAM_GB="$(python -c 'import torch; print(int(torch.cuda.get_device_properties(0).total_memory // 2**30) if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)"
ACTS_BATCH="${ACTS_BATCH:-$([ "$VRAM_GB" -ge 48 ] && echo 64 || echo 32)}"
DRIVERS_BATCH="${DRIVERS_BATCH:-$([ "$VRAM_GB" -ge 48 ] && echo 128 || echo 64)}"
up() {  # up <stage> <model> <tables>
    [ -n "$UPLOAD" ] && run "upload_$1" "$2" python scripts/12_release.py --tables "$3" --branch "$UPLOAD" &
}
[ $# -gt 0 ] || set -- Qwen/Qwen3-8B google/gemma-4-12B
for m in "$@"; do
    s="$(tr '[:upper:]' '[:lower:]' <<< "${m##*/}")"
    echo "$(date '+%F %T') model $m (acts batch $ACTS_BATCH, drivers batch $DRIVERS_BATCH)"
    hf download "$m" > /dev/null || { echo "$(date '+%F %T') FAILED download $m"; continue; }
    run "acts_$s" "$m" python scripts/06_activations.py --batch-docs "$ACTS_BATCH" && up "acts_$s" "$m" acts,docs &&
        run "firings_$s" "$m" python scripts/17_firings.py --save-minutes "${FIRINGS_SAVE_MINUTES:-30}" && up "firings_$s" "$m" firings,sources &&
        run "drivers_$s" "$m" python scripts/07_drivers.py --batch "$DRIVERS_BATCH" && up "drivers_$s" "$m" drivers || continue
    run "reads_$s" "$m" python scripts/08_reads.py && up "reads_$s" "$m" reads
    run "reads_at_firings_$s" "$m" python scripts/checks/reads_at_firings.py
    run "edges_$s" "$m" python scripts/16_edges.py && up "edges_$s" "$m" edges
    run "edges_context_$s" "$m" python scripts/checks/edges_in_context.py measure &&
        run "edges_context_score_$s" "$m" python scripts/checks/edges_in_context.py score
    run "causal_$s" "$m" python scripts/checks/causal_lens.py measure && up "files_$s" "$m" files &&
        run "causal_score_$s" "$m" python scripts/checks/causal_lens.py score --lenses record
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
