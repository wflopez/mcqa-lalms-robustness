import sys
sys.path.insert(1, sys.path[0].replace(sys.path[0].split('/')[-1], ''))

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

from utils.consistency_rate import calculate_consistency_rate, calculate_consistent_and_correct_rate


def main(args):

    # files to exclude
    exclude_files = [
        #"rephrased-a-with-d_gemma-3-12b-it",
        #"rephrased-a-with-qd_gemma-3-12b-it",
        #"rephrased-a-with-q_gemma-3-12b-it"
    ]

    # Get the results folder from the arguments
    results_folder = args.results_folder
    exp_folder = results_folder.rstrip('/').split('/')[-1]

    if not os.path.exists(results_folder):
        raise Exception(f"Results folder {results_folder} does not exist")

    if "mmau" in results_folder:
        benchmark = "mmau"
    elif "mmar" in results_folder:
        benchmark = "mmar"
    elif "mmsu" in results_folder:
        benchmark = "mmsu"
    else:
        raise Exception(f"Results folder {results_folder} does not include one of the benchmarks")

    print(f"Benchmark: {benchmark}")

    # List all JSON files in the outputs folder
    pattern = os.path.join(results_folder, "*.json")
    matching_files = glob.glob(pattern)

    if (len(matching_files) != 24) and ("perms_" in exp_folder):
        raise Exception(f"Expected 24 permutations but found {len(matching_files)} in {results_folder}")
    elif (len(matching_files) != 7) and ("question_rp" in exp_folder):
        raise Exception(f"Expected 7 permutations but found {len(matching_files)} in {results_folder}")
    elif (len(matching_files) != 7) and ("answer_rp" in exp_folder):
        raise Exception(f"Expected 7 permutations but found {len(matching_files)} in {results_folder}")

    # Load all results
    dicts = []
    for file in matching_files:
        if any(exclude in file for exclude in exclude_files):
            continue
        with open(file, "r") as f:
            results = json.load(f)
        dicts.append(results)

    # Read all results files and compute the mean accuracy
    total_accuracies = []
    categories_accuracies = []

    #print(dicts[0])


    for i in range(len(dicts)):
        if "mmau" in results_folder:
            # Read per task accuracy
            tasks_performance = dicts[i]["task"]
            categories_accuracies.append(tasks_performance)
        if "mmar" in results_folder:
            # Read per category accuracy
            modality_performance = dicts[i]["modality"]
            categories_accuracies.append(modality_performance)
        if "mmsu" in results_folder:
            # Read per modality accuracy
            modality_performance = dicts[i]["sub-sub-category"]
            categories_accuracies.append(modality_performance)
        total_accuracies.append(dicts[i]["total"])

    # Calculate mean, min, and max accuracies
    mean_acc = np.mean(total_accuracies)
    std_acc = np.std(total_accuracies)
    min_acc = np.min(total_accuracies)
    max_acc = np.max(total_accuracies)

    print(f"Results for {exp_folder} ({len(matching_files)} permutations):")
    print(f"Number of Permutations: {len(dicts)}")
    print("*"*30)
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Accuracy: {std_acc:.2f}%")
    print(f"Min Accuracy: {min_acc:.2f}%")
    print(f"Max Accuracy: {max_acc:.2f}%")
    print("*"*30)

    #################################

    # Now read all the outputs and compute consistency metrics
    print("Calculating consistency metrics...")
    perms_folder = results_folder.replace("/" + exp_folder, "/outputs/" + exp_folder)

    if not os.path.exists(perms_folder):
        raise Exception(f"Perm path {perms_folder} does not exist")

    pattern = os.path.join(perms_folder, "*.json")
    matching_files = glob.glob(pattern)

    if (len(matching_files) != 24) and ("perms_" in exp_folder):
        raise Exception(f"Expected 24 permutations but found {len(matching_files)} in {perms_folder}")
    elif (len(matching_files) != 7) and ("question_rp" in exp_folder):
        raise Exception(f"Expected 7 permutations but found {len(matching_files)} in {perms_folder}")
    elif (len(matching_files) != 7) and ("answer_rp" in exp_folder):
        raise Exception(f"Expected 7 permutations but found {len(matching_files)} in {perms_folder}")

    dicts = []
    
    # Load all outputs
    list_of_inferences_per_id = []
    list_of_clean_inferences_per_id = []
    list_of_gt_answers = []
    list_of_choices = []
    list_of_prompts = []

    for file in matching_files:
        if any(exclude in file for exclude in exclude_files):
            continue
        with open(file, "r") as f:
            results = json.load(f)
        dicts.append(results)
    
    n_ids = len(dicts[0])
    n_permutations = len(dicts)

    # Extra check to ensure that every json file has the same number of IDs
    for i in range(1, n_permutations):
        if len(dicts[i]) != n_ids:
            raise Exception(f"Mismatch in number of IDs: {len(dicts[i])} vs {n_ids} in file {matching_files[i]}")

    # Check if kimi
    output_key = 'model_output'

    for i in range(n_ids):
        inferences = []
        clean_inferences = []
        prompts = []
        choices = []
        answers = []

        #list_of_gt_answers.append(str(dicts[0][i]["answer"]))
        #list_of_choices.append(dicts[0][i]["choices"])
        #list_of_prompts.append(dicts[j][i]["prompt"])

        for j in range(n_permutations):
            prompts.append(dicts[j][i]["prompt"])
            choices.append(dicts[j][i]["choices"])
            answers.append(dicts[j][i]["answer"])

            model_output = dicts[j][i][output_key]

            clean_inference = model_output.lower()

            for prefix in ["(A) ", "(B) ", "(C) ", "(D) ", "(E) ", "(F) ", "(G) ", "(H) ", "(I) ",
                        "(a) ", "(b) ", "(c) ", "(d) ", "(e) ", "(f) ", "(g) ", "(h) ", "(i) "]:
                clean_inference = clean_inference.replace(prefix, "")

            inferences.append(model_output.strip().lower())
            clean_inferences.append(clean_inference.strip().lower())
        
        list_of_clean_inferences_per_id.append(clean_inferences)
        list_of_inferences_per_id.append(inferences)
        list_of_prompts.append(prompts)
        list_of_gt_answers.append(answers)
        list_of_choices.append(choices)

    # Compute consistency metrics
    cr = calculate_consistency_rate(list_of_clean_inferences_per_id)

    # Calculate Consistent and Correct Rate
    list_of_inferences_to_process = list_of_clean_inferences_per_id
    if benchmark == "mmsu":
        list_of_inferences_to_process = list_of_inferences_per_id

    ccr = calculate_consistent_and_correct_rate(
        responses=list_of_inferences_to_process,
        gt_answers=list_of_gt_answers,
        choices_list=list_of_choices,
        prompts_list=list_of_prompts,
        benchmark=benchmark
    )

    if benchmark == "mmsu":
        # ===== DEBUG BLOCK START - REMOVE THIS ENTIRE SECTION =====
        # Print final debug summary
        from utils.consistency_rate import print_final_debug_summary
        print_final_debug_summary()
        # ===== DEBUG BLOCK END =====

    # Generate a tsv file with the results
    output_file = os.path.join(results_folder, 'robustness.tsv')
    
    # Build a pandas DataFrame with experiment name, accuracy, consistency rate, and consistent and correct rate
    df = pd.DataFrame({
        'experiment': [exp_folder],
        'mean_accuracy': [mean_acc],
        'std_accuracy': [std_acc],
        'min_accuracy': [min_acc],
        'max_accuracy': [max_acc],
        'consistency_rate': [cr],
        'consistent_and_correct_rate': [ccr]
    })

    df.to_csv(output_file, sep='\t', index=False)

    # Print results
    print(f"Consistency Rate: {cr:.2f}")
    print(f"Consistent and Correct Rate: {ccr:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate permutation results.")
    parser.add_argument(
        '--results_folder',
        type=str,
        required=True,
        help='Path to the results folder with permutation results and outputs'
    )
    args = parser.parse_args()

    main(args)

