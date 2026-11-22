#!/usr/bin/env bash
# Ten-model network for the step 32 -> 64 (ideas list B1a), standard extra inputs (models'
# seasonal cycle, land mask, latitude), clean training inputs. The 256 x 512 grid needs
# batch 8 on the 32 GB card (peak about 29 GB).
#
#   scripts/r4.sh
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-60000}; BATCH=${BATCH:-8}; WORKERS=${WORKERS:-22}; NAME=${NAME:-std-r4}
MODELS=$(command ls data/proc/isimip_*.npz | xargs -n1 basename | sed 's/\.npz$//' | paste -sd, -)
mkdir -p runs/r4
LOG=runs/r4/r4.log
echo "=== $(date -Is) train $NAME (32 -> 64, batch $BATCH, $STEPS steps)" | tee -a $LOG
uv run python -m grace_diffusion.train --dataset "$MODELS" --kind twin --rungs 64 \
    --p-mean -0.4 --p-std 1.6 --sigma-max 20 --steps $STEPS --batch $BATCH --workers $WORKERS \
    --extras clim+land+lat --out "runs/$NAME" > "runs/r4/$NAME.stdout" 2>&1
echo "=== $(date -Is) $NAME exit $?: $(tail -n 1 "runs/$NAME/log.csv" 2>/dev/null)" | tee -a $LOG
echo "=== $(date -Is) r4 done" | tee -a $LOG
