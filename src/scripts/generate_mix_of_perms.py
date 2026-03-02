import json
import copy
import tqdm
import random
import itertools

import argparse
import numpy as np


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Generate JSON files with all permutations of models for benchmarks.")

    # probability of using rephrased question, answer, distractors
    args.add_argument("--p_rq", type=float, default=0.5, help="Probability of using rephrased question.")
    args.add_argument("--p_ra", type=float, default=0.5, help="Probability of using rephrased answer.")
    args.add_argument("--p_rd", type=float, default=0.5, help="Probability of using rephrased distractors.")
    args.add_argument("--p_choices_ordering", type=float, default=0.5, help="Probability of changing the ordering of the choices.")
    args.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility.")

    args = args.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    ################################################
    ############# Configuration ####################
    ################################################

    # For every benchmark, generate a json file with all possible permutations of the models
    benchmarks = ["MMAU", "MMAR", "MMSU"]

    # Isolated permutations to consider
    rephrased_questions = [
        "rephrased-q_gemma-3-12b-it",
        "rephrased-q-with-a_gemma-3-12b-it",
        "rephrased-q-with-da_gemini-2.5-flash-1",
        "rephrased-q-with-da_gemini-2.5-flash-2",
        "rephrased-q-with-da_gemini-2.5-flash-3",
        "rephrased-q-with-d_gemma-3-12b-it"
    ]

    rephrased_answers = [
        "rephrased-a-with-d_gemma-3-12b-it",
        "rephrased-a-with-qd_gemini-2.5-flash-1",
        "rephrased-a-with-qd_gemini-2.5-flash-2",
        "rephrased-a-with-qd_gemini-2.5-flash-3",
        "rephrased-a-with-qd_gemma-3-12b-it",
        "rephrased-a-with-q_gemma-3-12b-it"
    ]

    rephrased_distractors = [
        "rephrased-d-with-a_gemma-3-12b-it",
        "rephrased-d-with-qa_gemini-2.5-flash-1",
        "rephrased-d-with-qa_gemini-2.5-flash-2",
        "rephrased-d-with-qa_gemini-2.5-flash-3",
        "rephrased-d-with-qa_gemma-3-12b-it",
        "rephrased-d-with-q_gemma-3-12b-it"
    ]


    ################################################
    ############# Main script ######################
    ################################################


    for benchmark in benchmarks:

        if benchmark == "MMAU":
            dataset_path = "/mnt/matylda4/xlopezw00/MMAU-v05.15.25/"
            original_file = "mmau-test-mini.json"
        elif benchmark == "MMAR":
            dataset_path = "/mnt/matylda4/xlopezw00/MMAR/"
            original_file = "MMAR-meta.json"
        elif benchmark == "MMSU":
            dataset_path = "/mnt/matylda4/xlopezw00/MMSU/question/"
            original_file = "mmsu.json"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        with open(dataset_path + original_file, "r") as f:
            original_data = json.load(f)
        
        # Cache for loaded files to avoid repeated reading
        file_cache = {}
        
        all_permutations = []

        for item in tqdm.tqdm(original_data, desc=f"Processing {benchmark}"):
            new_item = copy.deepcopy(item) # Deep copy to avoid modifying the original item
            identifier = new_item["id"]

            permutations_applied = []

            ######################
            # REPHRASED QUESTIONS
            ######################

            # Use a probability to decide whether to use rephrased question or not
            if np.random.rand() < args.p_rq:
                rephrased_questions_to_use = random.choice(rephrased_questions) # Uniformly sample one of the rephrased questions
                
                # Read the file with the rephrased questions
                rq_filename = f"{dataset_path}{rephrased_questions_to_use}_{original_file}"
                
                # Use cache to avoid repeated file reads
                if rq_filename not in file_cache:
                    with open(rq_filename, "r") as f:
                        file_cache[rq_filename] = json.load(f)
                rq_data = file_cache[rq_filename]
                
                # Find the rephrased question with the same id
                for rq_item in rq_data:
                    if rq_item["id"] == identifier:
                        question = rq_item["question"]
                        break

                # Use this rephrased question
                new_item["question"] = question

                # Track permutation applied
                permutations_applied.append(f"Question rephrasing from {rephrased_questions_to_use}")
            
            ########################
            # REPHRASED CHOICES
            ########################

            # Use a probability to decide whether to use rephrased distractors or not
            if np.random.rand() < args.p_rd:
                rephrased_distractors_to_use = random.choice(rephrased_distractors) # Uniformly sample one of the rephrased distractors

                # Read the file with the rephrased distractors
                rd_filename = f"{dataset_path}{rephrased_distractors_to_use}_{original_file}"

                # Use cache to avoid repeated file reads
                if rd_filename not in file_cache:
                    with open(rd_filename, "r") as f:
                        file_cache[rd_filename] = json.load(f)
                rd_data = file_cache[rd_filename]

                # Find the rephrased distractor with the same id
                for rd_item in rd_data:
                    if rd_item["id"] == identifier:
                        choices = rd_item["choices"]
                        break

                # Use this rephrased distractors: answer remains the same
                new_item["choices"] = choices

                # Track permutation applied
                permutations_applied.append(f"Distractor rephrasing from {rephrased_distractors_to_use}")
            
            ######################
            # REPHRASED ANSWERS
            ######################

            # Use a probability to decide whether to use rephrased answer or not
            if np.random.rand() < args.p_ra:
                rephrased_answers_to_use = random.choice(rephrased_answers) # Uniformly sample one of the rephrased answers
                
                # Read the file with the rephrased answers
                ra_filename = f"{dataset_path}{rephrased_answers_to_use}_{original_file}"
                
                # Use cache to avoid repeated file reads
                if ra_filename not in file_cache:
                    with open(ra_filename, "r") as f:
                        file_cache[ra_filename] = json.load(f)
                ra_data = file_cache[ra_filename]
                
                # Find the rephrased answer with the same id
                for ra_item in ra_data:
                    if ra_item["id"] == identifier:
                        answer = ra_item["answer"]
                        break

                # Store previous answer
                previous_answer = new_item["answer"]

                # Handle edge cases for answers that might have changed formatting
                if "middle aged adult" in str(previous_answer):
                    previous_answer = previous_answer.replace("middle aged adult", "Middle-aged adult")
                elif "elderly adult" in str(previous_answer):
                    previous_answer = previous_answer.replace("elderly adult", "Elderly adult")

                # Use this rephrased answer
                new_item["answer"] = answer

                # Track permutation applied
                permutations_applied.append(f"Answer rephrasing from {rephrased_answers_to_use}")

                # Update in choices if previous answer was in choices, it should be corrected at this point
                # It should be there, but just in case. Every dataset has its missing answers...
                if previous_answer in new_item["choices"]:
                    choices = new_item["choices"]
                    # Replace previous answer with new answer
                    choices = [answer if choice == previous_answer else choice for choice in choices]
                    new_item["choices"] = choices
                else:
                    print(f"Warning: Previous answer '{previous_answer}' not found in choices for item ID '{identifier}'. The choices remain unchanged.")
                    # In this case, we insert the new answer into the choices if it's not already there
                    # insert in the first position replacing the first distractor
                    if answer not in new_item["choices"]:
                        new_item["choices"][0] = answer


            #######################
            # Choices ordering
            #######################

            # Use a probability to decide whether to change the ordering of the choices
            if np.random.rand() < args.p_choices_ordering:

                # Get the choices from the new item
                choices = new_item["choices"]

                # Calculate all possible permutations of the choices
                permutations = list(itertools.permutations(choices))

                # Remove the original order to ensure we actually change it
                permutations = [p for p in permutations if list(p) != choices]
                
                if permutations:
                    # Randomly select one permutation
                    selected_permutation = random.choice(permutations) # Uniformly sample one of the permutations
                    # Get the index of the selected permutation
                    selected_permutation_index = permutations.index(selected_permutation)

                    new_item["choices"] = list(selected_permutation)

                    # Track permutation applied
                    permutations_applied.append(f"Choices ordering: {selected_permutation_index} out of {len(permutations)}")
                else:
                    print(f"Warning: No alternative permutations available for item ID '{identifier}'. The choices remain unchanged. Choices: {choices}")

            #######################
            # Add new item to list of all permutations
            #######################
            new_item["permutations_applied"] = permutations_applied
            all_permutations.append(new_item)

        # Save all permutations to a new json file: "all_permutations_{original_file}"
        output_file = f"perm-mix-s-{args.seed}_{original_file}"

        with open(dataset_path + output_file, "w") as f:
            json.dump(all_permutations, f, indent=4)


