#!/usr/bin/env bash
# Stage 1 (design doc §7.3): the paired design at coarse scales, where GRACE can check it.
#   pre-train on WaterGAP alone, fine-tune on paired months (land-only GRACE in, WaterGAP
#   detail out), score on the paired validation months (2015-01 .. 2017-06).
#
#   scripts/stage1.sh [pretrain|finetune|eval|all]      (default: all)
#
# Runs are sequential (one 16->32 run fills the 32 GB card) and skipped if already finished.
set -u
cd "$(dirname "$0")/.."
STAGE=${1:-all}
IN=${IN:-grace_ddk5_land}          # input dataset of the paired samples
TGT=twin                           # WaterGAP v2.2e
PRE_STEPS=${PRE_STEPS:-60000}
PRE1_STEPS=${PRE1_STEPS:-40000}    # one-step 8->32: every batch is on the large grid
FT_STEPS=${FT_STEPS:-10000}
FT_LR=${FT_LR:-5e-5}
MEMBERS=${MEMBERS:-32}
NOISE="--p-mean -0.4 --p-std 1.6 --sigma-max 20"
mkdir -p runs/stage1 runs/eval
LOG=runs/stage1/stage1.log

done_steps() { [ -f "runs/$1/log.csv" ] && [ "$(tail -n 1 "runs/$1/log.csv" | cut -d, -f1)" = "$2" ]; }

train() {   # name steps args...
    local name=$1 steps=$2; shift 2
    if done_steps "$name" "$steps"; then echo "--- $name already done" | tee -a $LOG; return; fi
    echo "--- $(date -Is) train $name" | tee -a $LOG
    uv run python -m grace_diffusion.train --steps "$steps" --out "runs/$name" "$@" \
        > "runs/stage1/$name.stdout" 2>&1
    local rc=$?
    echo "--- $(date -Is) $name exit $rc: $(tail -n 1 "runs/$name/log.csv" 2>/dev/null)" | tee -a $LOG
}

if [ "$STAGE" = pretrain ] || [ "$STAGE" = all ]; then
    echo "=== $(date -Is) stage1 pretrain" | tee -a $LOG
    train pre-wg-r23 $PRE_STEPS --dataset $TGT --kind twin --rungs 16,32 $NOISE
    train pre-wg-r23-regress 40000 --dataset $TGT --kind twin --rungs 16,32 --model regress
    train pre-wg-8to32 $PRE1_STEPS --dataset $TGT --kind twin --rungs 8-32 $NOISE
fi

if [ "$STAGE" = finetune ] || [ "$STAGE" = all ]; then
    echo "=== $(date -Is) stage1 finetune" | tee -a $LOG
    FT="--dataset $IN --target $TGT --kind paired --lr $FT_LR --val-every 500"
    # B: two steps, WaterGAP pre-training
    train ft-wg-r23 $FT_STEPS $FT --rungs 16,32 $NOISE --init runs/pre-wg-r23/best.pt
    # B': two steps, ten-model pre-training (exists from G1)
    train ft-ten-r23 $FT_STEPS $FT --rungs 16,32 $NOISE --init runs/isimip10-r23/best.pt
    # C: one step 8 -> 32
    train ft-wg-8to32 $FT_STEPS $FT --rungs 8-32 $NOISE --init runs/pre-wg-8to32/best.pt
    # E: direct network
    train ft-wg-r23-regress $FT_STEPS $FT --rungs 16,32 --model regress \
        --init runs/pre-wg-r23-regress/best.pt
    # D: paired months only, no pre-training
    train paired-only-r23 30000 --dataset $IN --target $TGT --kind paired --rungs 16,32 $NOISE \
        --val-every 1000
fi

if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
    echo "=== $(date -Is) stage1 eval" | tee -a $LOG
    ev() {  # out-name steps [--target twin] ckpt specs...
        local out=$1 steps=$2; shift 2
        uv run python -m grace_diffusion.evaluate $IN paired val "$steps" --members $MEMBERS \
            --churn 20 --baselines --out "runs/eval/S1_$out.csv" "$@" 2>&1 | grep -v Warning | tee -a $LOG
    }
    ONE="--ckpt A=runs/pre-wg-r23/best.pt --ckpt B=runs/ft-wg-r23/best.pt
         --ckpt Bten=runs/ft-ten-r23/best.pt --ckpt D=runs/paired-only-r23/best.pt
         --ckpt E=runs/ft-wg-r23-regress/best.pt --ckpt Aten=runs/isimip10-r23/best.pt"
    TWO="--ckpt chainA=runs/pre-wg-r23/best.pt --ckpt chainB=runs/ft-wg-r23/best.pt
         --ckpt A1=runs/pre-wg-8to32/best.pt --ckpt C=runs/ft-wg-8to32/best.pt"
    # check 1: against WaterGAP's detail (the training target), held-out months
    ev wg_r23 16,32 --target $TGT $ONE
    ev wg_8to32 8-32 --target $TGT $TWO
    # check 2: against GRACE's own detail
    ev grace_r23 16,32 $ONE
    ev grace_8to32 8-32 $TWO
fi
echo "=== $(date -Is) stage1 $STAGE done" | tee -a $LOG
