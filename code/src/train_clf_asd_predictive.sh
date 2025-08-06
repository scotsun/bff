#!/usr/bin/bash

data_usage_values=(0.1 0.15 0.3 0.85)
modalityCheckpoint_values=("first_check" "mid_check")

run_expr() {
    python train_clf_asd_predictive.py --modalityCheckpoint "$1" \
        --data_usage "$2" \
        --device cuda:0
}

for checkpoint in "${modalityCheckpoint_values[@]}"; do
    for f in "${data_usage_values[@]}"; do
        run_expr "$checkpoint" "$f"
    done
done