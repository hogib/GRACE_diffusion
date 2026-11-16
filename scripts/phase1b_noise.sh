#!/usr/bin/env bash
# Phase 1b: retrain the R1+R2 diffusion model with more high-noise training.
# Diagnosis (scripts/diagnose_sampler.py, 2026-11-16): with EDM's default ln(sigma) ~ N(-1.2, 1.2^2)
# only ~0.2% of batches have sigma > 10, and the R1+R2 model reads its input weakly there
# (denoiser corr 0.26-0.40 vs regress 0.89-0.92); the R3 oracle, with 4-16x more pixels per
# batch, did learn it (0.79). Two variants, both sampled from sigma_max = 20:
#   wide  ln(sigma) ~ N(-0.4, 1.6^2)
#   logu  ln(sigma) uniform on [0.002, 20]
#   nohup setsid bash scripts/phase1b_noise.sh > runs/phase1b.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-60000}
WORKERS=${WORKERS:-5}
MEMBERS=${MEMBERS:-32}
echo "=== $(date -Is) phase1b start, commit $(git rev-parse --short HEAD), steps $STEPS"

train() {
    local name=$1; shift
    mkdir -p "runs/$name"
    echo "--- $(date -Is) train $name $*"
    uv run python -m grace_diffusion.train --out "runs/$name" --steps "$STEPS" \
        --workers "$WORKERS" --dataset grace_ddk5 --kind grace --rungs 8,16 --model diffusion \
        --sigma-max 20 "$@" > "runs/$name/stdout.log" 2>&1
    local rc=$?
    echo "--- $(date -Is) done $name (exit $rc)"
}
train diffusion-ddk5-r12-wide --noise lognormal --p-mean -0.4 --p-std 1.6 &
train diffusion-ddk5-r12-logu --noise loguniform &
wait

W=runs/diffusion-ddk5-r12-wide/best.pt
U=runs/diffusion-ddk5-r12-logu/best.pt
for c in $W $U; do
    echo "--- $(date -Is) diagnose $c"
    uv run python scripts/diagnose_sampler.py $c runs/regress-ddk5-r12/best.pt \
        --rungs 8,16,32 --members 16 --steps 32 2>&1 | grep -v Warning
done
mkdir -p runs/eval
echo "--- $(date -Is) evaluate A"
uv run python -m grace_diffusion.evaluate grace_ddk5 grace val 16 --members "$MEMBERS" \
    --ckpt wide=$W --ckpt logu=$U --out runs/eval/A1b_grace_ddk5.csv
for ds in grace_ddk5 grace_raw; do
    echo "--- $(date -Is) evaluate B $ds"
    uv run python -m grace_diffusion.evaluate $ds grace val 32 --members "$MEMBERS" \
        --ckpt wide=$W --ckpt logu=$U --out runs/eval/B1b_$ds.csv
done
echo "=== $(date -Is) phase1b done"
