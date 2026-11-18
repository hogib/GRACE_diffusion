#!/usr/bin/env bash
# G1 spread fix on the existing ten-model network: churn and inflation, fitted on
# (a) GRACE training years (the network never saw GRACE) and (b) the models' held-out years.
set -u
cd "$(dirname "$0")/.."
CK=runs/isimip10-r23/best.pt
MODELS=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | paste -sd, -)
mkdir -p runs/g1
echo "=== $(date -Is) calibrate on GRACE land training years"
uv run python scripts/calibrate_spread.py $CK --dataset grace_ddk5_land --kind grace \
  --split train --rungs 16,32 --months 36 --stride 4 --churn 0,5,20 \
  --out runs/eval/G1cal_grace_train.csv
echo "=== $(date -Is) calibrate on held-out model years"
uv run python scripts/calibrate_spread.py $CK --dataset "$MODELS" --kind twin \
  --split val --rungs 16,32 --months 4 --stride 6 --churn 0,5,20 \
  --out runs/eval/G1cal_models_val.csv
echo "=== $(date -Is) calibrate done"
