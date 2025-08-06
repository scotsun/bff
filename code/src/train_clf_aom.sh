#!/usr/bin/bash

mixingModule_values=("softmax-gating" "self-attention" "masked-avg")
modalityCheckpoint_values=("first_check" "mid_check" "final_check")

for mixing in "${mixingModule_values[@]}"; do
    python train_clf_aom.py --modalities prenatal --mixingModule "$mixing" --task binary --device cuda:2
done
for mixing in "${mixingModule_values[@]}"; do
    python train_clf_aom.py --modalities prenatal birth --mixingModule "$mixing" --task binary --device cuda:2
done
for mixing in "${mixingModule_values[@]}"; do
    python train_clf_aom.py --modalities all --mixingModule "$mixing" --task binary --device cuda:2
done

run_expr() {
    python train_clf_aom.py --mixingModule "$1" \
        --modalityCheckpoint "$2" \
        "${@:3}" \
        --task binary \
        --device cuda:2
}

for mixing in "${mixingModule_values[@]}"; do
    for checkpoint in "${modalityCheckpoint_values[@]}"; do
        run_expr "$mixing" "$checkpoint"
        run_expr "$mixing" "$checkpoint" --addContrastive --zpOnly
        run_expr "$mixing" "$checkpoint" --addContrastive
    done
done