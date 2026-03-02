#!/bin/bash

#PERMUTATIONS_UNDER_TEST=("perms" "question_rp" "distractors_rp" "answer_rp")
#MODEL_NAMES=("af2" "af3" "Qwen2.5-Omni-7B" "kimi-audio")
#DATASETS=("mmau-v05.15.25" "mmar" "mmsu")
PERMUTATIONS_UNDER_TEST=("distractors_rp" "perm-mix")
MODEL_NAMES=("af2" "af3" "Qwen2.5-Omni-7B" "kimi-audio")
DATASETS=("mmau-v05.15.25" "mmar" "mmsu")


for PERMUTATION_UNDER_TEST in "${PERMUTATIONS_UNDER_TEST[@]}"; do
    echo "=== Processing permutation type: $PERMUTATION_UNDER_TEST ==="

    for dataset in "${DATASETS[@]}"; do
        for model_name in "${MODEL_NAMES[@]}"; do
            perm_results_dir="results/${dataset}/${PERMUTATION_UNDER_TEST}_${model_name}"

            echo "Processing dataset: $dataset with model: $model_name"
            echo "Results directory: $perm_results_dir"

            python src/scripts/calculate_robustness_results.py \
            --results_folder "$perm_results_dir"

            echo -e "-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-\n"

        done
    done
done