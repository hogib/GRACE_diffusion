#!/usr/bin/env bash
# Paired experiment, short first pass (design doc Appendix B.5): does fine-tuning on real GRACE help, and how much
# does the GRACE / WaterGAP amplitude mismatch hurt? Uses the existing ten-model network.
#   B' : fine-tune runs/isimip10-r23 on paired months     D : paired months only, from scratch
set -u
cd "$(dirname "$0")/.."
IN=${IN:-grace_ddk5_land}; TGT=twin; MEMBERS=${MEMBERS:-32}
NOISE="--p-mean -0.4 --p-std 1.6 --sigma-max 20"
mkdir -p runs/stage1 runs/eval
LOG=runs/stage1/short.log
train() {
    local name=$1 steps=$2; shift 2
    if [ -f "runs/$name/log.csv" ] && [ "$(tail -n 1 "runs/$name/log.csv" | cut -d, -f1)" = "$steps" ]; then
        echo "--- $name already done" | tee -a $LOG; return; fi
    echo "--- $(date -Is) train $name" | tee -a $LOG
    uv run python -m grace_diffusion.train --steps "$steps" --out "runs/$name" "$@" \
        > "runs/stage1/$name.stdout" 2>&1
    local rc=$?
    echo "--- $(date -Is) $name exit $rc: $(tail -n 1 "runs/$name/log.csv" 2>/dev/null)" | tee -a $LOG
}
echo "=== $(date -Is) stage1 short pass" | tee -a $LOG
train ft-ten-r23 10000 --dataset $IN --target $TGT --kind paired --rungs 16,32 $NOISE \
    --lr 5e-5 --val-every 500 --init runs/isimip10-r23/best.pt
train paired-only-r23 30000 --dataset $IN --target $TGT --kind paired --rungs 16,32 $NOISE \
    --val-every 1000
CK="--ckpt Aten=runs/isimip10-r23/best.pt --ckpt Bten=runs/ft-ten-r23/best.pt
    --ckpt D=runs/paired-only-r23/best.pt"
ev() {
    local out=$1; shift
    echo "--- $(date -Is) evaluate $out" | tee -a $LOG
    uv run python -m grace_diffusion.evaluate $IN paired val 16,32 --members $MEMBERS --churn 20 \
        --baselines --out "runs/eval/S1short_$out.csv" "$@" $CK 2>&1 | grep -v Warning | tee -a $LOG
}
ev wg --target $TGT        # check 1: against WaterGAP's detail, held-out months
ev grace                   # check 2: against GRACE's own detail
echo "=== $(date -Is) stage1 short done" | tee -a $LOG
