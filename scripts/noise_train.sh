#!/usr/bin/env bash
# Training with GRACE-like input noise (ideas list B4). Ten-model network at the coarse steps
# with the standard extra inputs (models' seasonal cycle, land mask, latitude).
#
#   std-clean   standard inputs, clean training inputs           (the reference)
#   std-noise   standard inputs, input noise up to 2x formal errors
#   full-noise  standard inputs + the models' fields for the month, input noise
#
#   scripts/noise_train.sh [train|eval|all] [variants...]   default variants: std-clean std-noise
set -u
cd "$(dirname "$0")/.."
STAGE=${1:-all}; shift || true
VARIANTS=${*:-std-clean std-noise}
STEPS=${STEPS:-60000}; MEMBERS=${MEMBERS:-32}; WORKERS=${WORKERS:-16}; FMAX=${FMAX:-2}
MODELS=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | paste -sd, -)
mkdir -p runs/noise runs/eval
LOG=runs/noise/noise.log
args() { case $1 in
    std-clean)  echo "--extras clim+land+lat";;
    std-noise)  echo "--extras clim+land+lat --input-noise $FMAX";;
    full-noise) echo "--extras models+clim+land+lat --input-noise $FMAX";;
    *) echo "unknown variant $1" >&2; exit 1;; esac; }

if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
    echo "=== $(date -Is) noise train: $VARIANTS" | tee -a $LOG
    for v in $VARIANTS; do
        name=$v
        if [ -f "runs/$name/log.csv" ] && [ "$(tail -n 1 "runs/$name/log.csv" | cut -d, -f1)" = "$STEPS" ]; then
            echo "--- $name already done" | tee -a $LOG; continue; fi
        echo "--- $(date -Is) train $name ($(args $v))" | tee -a $LOG
        uv run python -m grace_diffusion.train --dataset "$MODELS" --kind twin --rungs 16,32 \
            --p-mean -0.4 --p-std 1.6 --sigma-max 20 --steps $STEPS --workers $WORKERS \
            $(args $v) --out "runs/$name" > "runs/noise/$name.stdout" 2>&1
        rc=$?
        echo "--- $(date -Is) $name exit $rc: $(tail -n 1 "runs/$name/log.csv" 2>/dev/null)" | tee -a $LOG
    done
fi

if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
    CK="--ckpt none=runs/isimip10-r23/best.pt --ckpt all=runs/wall-all/best.pt"
    for v in $VARIANTS; do [ -f "runs/$v/best.pt" ] && CK="$CK --ckpt $v=runs/$v/best.pt"; done
    ev() {  # out-name kind
        echo "--- $(date -Is) evaluate $1" | tee -a $LOG
        uv run python -m grace_diffusion.evaluate grace_ddk5_land "$2" val 16,32 --members $MEMBERS \
            --churn 20 --out "runs/eval/noisetrain_$1.csv" $CK 2>&1 | grep -v "Warning\|months in" | tee -a $LOG
    }
    ev 2015 paired
    ev 2020 grace
fi
echo "=== $(date -Is) noise $STAGE done" | tee -a $LOG
