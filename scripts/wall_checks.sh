#!/usr/bin/env bash
# Two checks on the extra-input networks (design doc B.7).
#  1. Inside the held-out Türkiye box, on model data (truth known): do networks with
#     geographic inputs behave worse there than outside?
#  2. Spread calibration of the 'all' network: inflation fitted on GRACE 2002-2014 and on the
#     models' held-out years.
set -u
cd "$(dirname "$0")/.."
LOG=runs/wall/checks.log
CK="--ckpt none=runs/isimip10-r23/best.pt --ckpt geo=runs/wall-geo/best.pt --ckpt all=runs/wall-all/best.pt"
echo "=== $(date -Is) check 1: Türkiye box on model data" | tee -a $LOG
for m in isimip_watergap2-2e isimip_h08; do
  for region in val_land test_box; do
    echo "--- $m $region" | tee -a $LOG
    uv run python -m grace_diffusion.evaluate $m twin val 16,32 --members 16 --churn 20 --baselines \
        --max-months 18 --region $region --out runs/eval/box_${m}_${region}.csv $CK 2>&1 \
        | grep -v "Warning\|months in" | tee -a $LOG
  done
done
echo "=== $(date -Is) check 2: spread of the 'all' network" | tee -a $LOG
MODELS=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | paste -sd, -)
echo "--- fitted on GRACE 2002-2014" | tee -a $LOG
uv run python scripts/calibrate_spread.py runs/wall-all/best.pt --dataset grace_ddk5_land --kind paired \
    --split train --rungs 16,32 --months 35 --stride 4 --churn 20 \
    --out runs/eval/cal_all_grace.csv 2>&1 | grep -v "Warning\|months in" | tee -a $LOG
echo "--- fitted on held-out model years" | tee -a $LOG
uv run python scripts/calibrate_spread.py runs/wall-all/best.pt --dataset "$MODELS" --kind twin \
    --split val --rungs 16,32 --months 4 --stride 6 --churn 20 \
    --out runs/eval/cal_all_models.csv 2>&1 | grep -v "Warning\|months in" | tee -a $LOG
echo "=== $(date -Is) checks done" | tee -a $LOG
