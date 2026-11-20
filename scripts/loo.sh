#!/usr/bin/env bash
# Leave one model out at the coarse steps (design doc §7.1 level 3, previewed where GRACE can
# also grade). Standard extra inputs (models' seasonal cycle, land mask, latitude), clean
# training inputs. For each withheld model: train on the other nine, score on the withheld
# model's held-out years, next to the ten-model network (runs/std-clean), which has seen it.
#
#   scripts/loo.sh [train|eval|all] [models...]     default: watergap2-2e jules-w2
set -u
cd "$(dirname "$0")/.."
STAGE=${1:-all}; shift || true
HELD=${*:-watergap2-2e jules-w2}
STEPS=${STEPS:-60000}; MEMBERS=${MEMBERS:-32}; WORKERS=${WORKERS:-16}
mkdir -p runs/loo runs/eval
LOG=runs/loo/loo.log
for h in $HELD; do
    name=loo-$h
    REST=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | grep -v "^isimip_$h$" | paste -sd, -)
    if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
        if [ -f "runs/$name/log.csv" ] && [ "$(tail -n 1 "runs/$name/log.csv" | cut -d, -f1)" = "$STEPS" ]; then
            echo "--- $name already done" | tee -a $LOG
        else
            echo "--- $(date -Is) train $name (without isimip_$h)" | tee -a $LOG
            uv run python -m grace_diffusion.train --dataset "$REST" --kind twin --rungs 16,32 \
                --p-mean -0.4 --p-std 1.6 --sigma-max 20 --steps $STEPS --workers $WORKERS \
                --extras clim+land+lat --bank-exclude isimip_$h --out "runs/$name" \
                > "runs/loo/$name.stdout" 2>&1
            echo "--- $(date -Is) $name exit $?: $(tail -n 1 "runs/$name/log.csv" 2>/dev/null)" | tee -a $LOG
        fi
    fi
    if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
        echo "--- $(date -Is) evaluate on isimip_$h (held-out years)" | tee -a $LOG
        uv run python -m grace_diffusion.evaluate isimip_$h twin val 16,32 --members $MEMBERS --churn 20 \
            --baselines --max-months 36 --ckpt unseen=runs/$name/best.pt --ckpt seen=runs/std-clean/best.pt \
            --out runs/eval/loo_$h.csv 2>&1 | grep -v "Warning\|months in" | tee -a $LOG
        echo "--- $(date -Is) evaluate $name on GRACE" | tee -a $LOG
        uv run python -m grace_diffusion.evaluate grace_ddk5_land paired val 16,32 --members $MEMBERS \
            --churn 20 --ckpt nine=runs/$name/best.pt --out runs/eval/loo_${h}_grace.csv 2>&1 \
            | grep -v "Warning\|months in" | tee -a $LOG
    fi
done
echo "=== $(date -Is) loo $STAGE done" | tee -a $LOG
