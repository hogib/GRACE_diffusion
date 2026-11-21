#!/usr/bin/env bash
# Scoring of the noise-trained networks (design doc B.9) with perturbed GRACE inputs at
# sampling time (B.8): formal errors times 1 and 2, on both validation periods. No training.
# Waits until the GPU has MINFREE MiB free before each evaluation (the card is shared).
#
#   scripts/noise_train_eval.sh
set -u
cd "$(dirname "$0")/.."
MEMBERS=${MEMBERS:-32}; CHUNK=${CHUNK:-16}; MINFREE=${MINFREE:-8000}
mkdir -p runs/noise runs/eval
LOG=runs/noise/noise_eval.log
NEW="--ckpt std-clean=runs/std-clean/best.pt --ckpt std-noise=runs/std-noise/best.pt --ckpt full-noise=runs/full-noise/best.pt"
OLD="--ckpt none=runs/isimip10-r23/best.pt --ckpt all=runs/wall-all/best.pt"
wait_free() {
    while :; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
        [ "$free" -ge "$MINFREE" ] && return
        echo "    $(date -Is) waiting: $free MiB free" >> $LOG
        sleep 120
    done
}
ev() {  # out-name kind factor checkpoints
    wait_free
    echo "--- $(date -Is) evaluate $1, noise factor $3" | tee -a $LOG
    uv run python -m grace_diffusion.evaluate grace_ddk5_land "$2" val 16,32 --members $MEMBERS \
        --chunk $CHUNK --churn 20 --input-noise "$3" --out "runs/eval/noisetrain_$1_f$3.csv" $4 2>&1 \
        | grep -v "Warning\|months in\|nanmean" | tee -a $LOG
}
echo "=== $(date -Is) noise-trained networks with perturbed inputs" | tee -a $LOG
for f in 1 2; do
    ev 2015 paired $f "$NEW"
    ev 2020 grace $f "$OLD $NEW"
done
echo "=== $(date -Is) noise eval done" | tee -a $LOG
