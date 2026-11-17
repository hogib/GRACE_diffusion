#!/usr/bin/env bash
# Phase 1c: the twin with the fixed noise schedule (Phase 1b 'wide'). Separates "GRACE fields
# are not self-similar across octaves" from "GRACE noise / DDK attenuation at 16-32 breaks
# transfer": the twin has neither. Scores R2 (in-distribution), R3 and R4 (transfer).
#   nohup setsid bash scripts/phase1c_twin.sh > runs/phase1c.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-60000}
echo "=== $(date -Is) phase1c start, commit $(git rev-parse --short HEAD), steps $STEPS"
name=diffusion-twin-r12-wide
mkdir -p runs/$name runs/eval
uv run python -m grace_diffusion.train --out runs/$name --steps "$STEPS" --workers 5 \
    --dataset twin --kind twin --rungs 8,16 --model diffusion \
    --noise lognormal --p-mean -0.4 --p-std 1.6 --sigma-max 20 > runs/$name/stdout.log 2>&1
echo "--- $(date -Is) done $name (exit $?)"
uv run python scripts/diagnose_sampler.py runs/$name/best.pt --dataset twin --kind twin \
    --rungs 8,16,32,64 --members 16 --steps 32 2>&1 | grep -v Warning
uv run python -m grace_diffusion.evaluate twin twin val 16,32,64 --members 32 --chunk 8 \
    --ckpt wide=runs/$name/best.pt --out runs/eval/C1c_twin.csv
echo "=== $(date -Is) phase1c done"
