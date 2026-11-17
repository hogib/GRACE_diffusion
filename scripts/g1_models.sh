#!/usr/bin/env bash
# G1 (design doc 01, section 5): train on ten ISIMIP3a hydrological models at the scales GRACE
# still resolves (8->16, 16->32), apply to land-only GRACE, score against GRACE's own detail.
# Reference: the same network trained on land-only GRACE (the fair upper bound).
#   nohup setsid bash scripts/g1_models.sh > runs/g1/g1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-60000}
WORKERS=${WORKERS:-5}
MEMBERS=${MEMBERS:-32}
MODELS=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | paste -sd,)
echo "=== $(date -Is) g1 start, commit $(git rev-parse --short HEAD), steps $STEPS"
echo "models: $MODELS"

train() {
    local name=$1; shift
    mkdir -p "runs/$name"
    echo "--- $(date -Is) train $name $*"
    uv run python -m grace_diffusion.train --out "runs/$name" --steps "$STEPS" \
        --workers "$WORKERS" --rungs 16,32 --model diffusion \
        --noise lognormal --p-mean -0.4 --p-std 1.6 --sigma-max 20 "$@" > "runs/$name/stdout.log" 2>&1
    local rc=$?
    echo "--- $(date -Is) done $name (exit $rc)"
}
train isimip10-r23   --dataset "$MODELS" --kind twin &
train graceland-r23  --dataset grace_ddk5_land --kind grace &
wait

M=runs/isimip10-r23/best.pt
G=runs/graceland-r23/best.pt
for ds in grace_ddk5_land grace_raw_land; do
    echo "--- $(date -Is) evaluate G1 $ds"
    uv run python -m grace_diffusion.evaluate $ds grace val 16,32 --members "$MEMBERS" \
        --baselines --ckpt models=$M --ckpt grace-land=$G --out runs/eval/G1_$ds.csv
done
echo "--- $(date -Is) evaluate G1 grace_ddk5 (full maps, models network only)"
uv run python -m grace_diffusion.evaluate grace_ddk5 grace val 16,32 --members "$MEMBERS" \
    --ckpt models=$M --out runs/eval/G1_grace_ddk5_full.csv
echo "=== $(date -Is) g1 done"
