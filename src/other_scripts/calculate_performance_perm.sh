#!/bin/bash


#PERMUATION_UNDER_TEST="perms"
#PERMUATION_UNDER_TEST="question_rp"
#PERMUATION_UNDER_TEST="answer_rp"
#PERMUATION_UNDER_TEST="distractors_rp"
PERMUATION_UNDER_TEST="perm-mix"

#MODEL_NAMES=("af2" "af3" "Qwen2.5-Omni-7B" "kimi-audio")
DATASETS=("mmau-v05.15.25" "mmar" "mmsu")
MODEL_NAMES=("af2" "af3" "Qwen2.5-Omni-7B" "kimi-audio")

for dataset in "${DATASETS[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        perm_output_dir="results/${dataset}/outputs/${PERMUATION_UNDER_TEST}_${model_name}"
        perm_results_dir="results/${dataset}/${PERMUATION_UNDER_TEST}_${model_name}"

        echo "Processing dataset: $dataset with model: $model_name"
        echo "Output directory: $perm_output_dir"
        echo "Results directory: $perm_results_dir"

        mkdir -p "$perm_results_dir"

        # Check if "mmau", "mmar", or "mmsu" is in the name, depending on that execute different script
        if [[ "$perm_output_dir" == *"mmau"* ]]; then
            echo "Using MMAU dataset"
            script_to_run="src/evaluate_mmau.py"
        elif [[ "$perm_output_dir" == *"mmar"* ]]; then
            echo "Using MMAR dataset"
            script_to_run="src/evaluate_mmar.py"
        elif [[ "$perm_output_dir" == *"mmsu"* ]]; then
            echo "Using MMSU dataset"
            script_to_run="src/evaluate_mmsu.py"
        else
            echo "Unknown dataset in output directory: $perm_output_dir"
            exit 1
        fi

        for file in "$perm_output_dir"/*; do
            if [ -f "$file" ]; then
                echo "File: $file"
                python $script_to_run \
                --input "$file" \
                --dest_folder "$perm_results_dir"
            fi
        done

    done
done
