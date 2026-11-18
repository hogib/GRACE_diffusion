#!/usr/bin/env bash
# "Wall" experiments (2026-11-18): the ten-model network at the coarse steps (8->16, 16->32),
# with extra input channels, scored against GRACE's own detail. Which additions help?
#
#   hist    previous months of the blurred field (lags 1, 2, 3, 6, 12)
#   models  the other models' detail for the same month (needs model runs: up to 2019)
#   clim    the models' mean seasonal cycle of the detail (available for any month)
#   geo     land mask and latitude
#   all     everything
#
#   scripts/wall.sh [train|eval|all] [variants...]     e.g.  scripts/wall.sh train hist clim
#
# Runs are sequential (one 16->32 run fills the 32 GB card) and skipped if already finished.
# Reference without extras: runs/isimip10-r23 (18.6 / 33.0 mm on 2015-2017, 22.6 / 29.8 on 2020-2022).
set -u
cd "$(dirname "$0")/.."
STAGE=${1:-all}; shift || true
VARIANTS=${*:-hist models clim geo all}
STEPS=${STEPS:-60000}; MEMBERS=${MEMBERS:-32}; WORKERS=${WORKERS:-16}
MODELS=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | paste -sd, -)
mkdir -p runs/wall runs/eval
LOG=runs/wall/wall.log
spec() { case $1 in
    hist) echo "hist:1,2,3,6,12";; models) echo "models";; clim) echo "clim";;
    geo) echo "land+lat";; all) echo "hist:1,2,3,6,12+models+clim+land+lat";;
    *) echo "unknown variant $1" >&2; exit 1;; esac; }

if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
    echo "=== $(date -Is) wall train: $VARIANTS" | tee -a $LOG
    for v in $VARIANTS; do
        name=wall-$v
        if [ -f "runs/$name/log.csv" ] && [ "$(tail -n 1 "runs/$name/log.csv" | cut -d, -f1)" = "$STEPS" ]; then
            echo "--- $name already done" | tee -a $LOG; continue; fi
        echo "--- $(date -Is) train $name ($(spec $v))" | tee -a $LOG
        uv run python -m grace_diffusion.train --dataset "$MODELS" --kind twin --rungs 16,32 \
            --p-mean -0.4 --p-std 1.6 --sigma-max 20 --steps $STEPS --workers $WORKERS \
            --extras "$(spec $v)" --out "runs/$name" > "runs/wall/$name.stdout" 2>&1
        rc=$?
        echo "--- $(date -Is) $name exit $rc: $(tail -n 1 "runs/$name/log.csv" 2>/dev/null)" | tee -a $LOG
    done
fi

if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
    CK="--ckpt none=runs/isimip10-r23/best.pt"
    for v in $VARIANTS; do [ -f "runs/wall-$v/best.pt" ] && CK="$CK --ckpt $v=runs/wall-$v/best.pt"; done
    ev() {  # out-name kind
        echo "--- $(date -Is) evaluate $1" | tee -a $LOG
        uv run python -m grace_diffusion.evaluate grace_ddk5_land "$2" val 16,32 --members $MEMBERS \
            --churn 20 --baselines --out "runs/eval/wall_$1.csv" $CK 2>&1 | grep -v Warning | tee -a $LOG
    }
    ev 2015 paired     # 2015-01 .. 2017-06: the models' fields for the month exist
    ev 2020 grace      # 2020 .. 2022: no model runs, the 'models' channel is empty
fi
echo "=== $(date -Is) wall $STAGE done" | tee -a $LOG
