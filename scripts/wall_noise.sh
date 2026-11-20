#!/usr/bin/env bash
# Spread from input uncertainty (ideas list C2), at sampling time, no training: each member's
# GRACE input is perturbed with the month's formal errors times a factor; only the generated
# detail is kept. Scan of the factor on fitting months (2002-2014) and validation months
# (2015-01 .. 2017-06), for the all-inputs network and the plain one.
set -u
cd "$(dirname "$0")/.."
LOG=runs/wall/noise.log
echo "=== $(date -Is) input-noise scan" | tee -a $LOG
[ -f data/proc/grace_sigma.npz ] || uv run python -m grace_diffusion.prep data/raw data/proc grace_sigma
for f in 0 1 2 4 8; do
  for split in train val; do
    [ $split = train ] && S="--stride 4" || S=""
    echo "--- factor $f, $split" | tee -a $LOG
    uv run python -m grace_diffusion.evaluate grace_ddk5_land paired $split 16,32 --members 32 \
        --churn 20 --input-noise $f $S --ckpt all=runs/wall-all/best.pt \
        --ckpt none=runs/isimip10-r23/best.pt --out runs/eval/noise_${split}_f$f.csv 2>&1 \
        | grep -v "Warning\|months in\|nanmean" | tee -a $LOG
  done
done
echo "=== $(date -Is) input-noise scan done" | tee -a $LOG
