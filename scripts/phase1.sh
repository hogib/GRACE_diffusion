#!/usr/bin/env bash
# Phase 1 (design doc, Appendix B.1-B.2): train the four models in parallel, then score experiments A, B, C.
#
# Run from anywhere on the GPU server, detached:
#   nohup setsid bash scripts/phase1.sh [STAGE] > runs/phase1.log 2>&1 &
# STAGE: all (default) | oracle (oracle + scoring) | eval (scoring only).
# Overrides: STEPS (default 40000), WORKERS per run (default 5), MEMBERS (default 32).
# The oracle trains alone after the other three: its 128x256 grids do not fit
# on a 32 GB GPU next to three R1+R2 runs.
#
# Training runs:
#   diffusion-ddk5-r12   diffusion, GRACE DDK5, rungs R1+R2 (4->8, 8->16)   the gate model
#   regress-ddk5-r12     deterministic U-Net, same data                       baseline
#   oracle-ddk5-r3       diffusion trained on R3 itself (16->32)              cost of transfer
#   diffusion-twin-r12   diffusion, WaterGAP twin, R1+R2                       closed loop to R5
# Scoring (validation months only; the test split stays untouched until the gate decision):
#   A  R2, GRACE DDK5: baselines, diffusion, regress
#   B  R3, GRACE DDK5 and unfiltered: baselines, diffusion and regress (transferred), oracle
#   C  twin R2-R5: baselines, twin diffusion (R4, R5 are two and three octaves beyond training)
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:-all}
STEPS=${STEPS:-40000}
WORKERS=${WORKERS:-5}
MEMBERS=${MEMBERS:-32}

echo "=== $(date -Is) phase1 $STAGE start, commit $(git rev-parse --short HEAD), steps $STEPS"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

train() {
    local name=$1; shift
    mkdir -p "runs/$name"
    echo "--- $(date -Is) train $name $*"
    uv run python -m grace_diffusion.train --out "runs/$name" --steps "$STEPS" \
        --workers "$WORKERS" "$@" > "runs/$name/stdout.log" 2>&1
    local rc=$?
    echo "--- $(date -Is) done $name (exit $rc)"
}

if [ "$STAGE" = all ]; then
    train diffusion-ddk5-r12 --dataset grace_ddk5 --kind grace --rungs 8,16 --model diffusion &
    train regress-ddk5-r12   --dataset grace_ddk5 --kind grace --rungs 8,16 --model regress &
    train diffusion-twin-r12 --dataset twin       --kind twin  --rungs 8,16 --model diffusion &
    wait
fi
if [ "$STAGE" = all ] || [ "$STAGE" = oracle ]; then
    train oracle-ddk5-r3 --dataset grace_ddk5 --kind grace --rungs 32 --model diffusion
fi

mkdir -p runs/eval
ev() {
    echo "--- $(date -Is) evaluate $*"
    uv run python -m grace_diffusion.evaluate "$@"
}
D=runs/diffusion-ddk5-r12/best.pt
R=runs/regress-ddk5-r12/best.pt
O=runs/oracle-ddk5-r3/best.pt
T=runs/diffusion-twin-r12/best.pt

ev grace_ddk5 grace val 16 --members "$MEMBERS" --baselines \
    --ckpt diffusion=$D --ckpt regress=$R --out runs/eval/A_grace_ddk5.csv
for ds in grace_ddk5 grace_raw; do
    ev $ds grace val 32 --members "$MEMBERS" --baselines \
        --ckpt diffusion=$D --ckpt regress=$R --ckpt oracle=$O --out runs/eval/B_$ds.csv
done
ev twin twin val 16,32,64 --members "$MEMBERS" --baselines --ckpt diffusion=$T \
    --out runs/eval/C_twin.csv
ev twin twin val 128 --members 16 --chunk 4 --baselines --ckpt diffusion=$T \
    --out runs/eval/C_twin_r5.csv

echo "=== $(date -Is) phase1 done"
