import os
import re
import json
import glob
import tqdm

import pandas as pd
import numpy as np


def mmau_string_match(prediction, answer, choices, prompts) -> (bool, int):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', str(text).lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    # Get the chosen position based on the content regex
    chosed_position = -1
    for idx, choice in enumerate(choices):
        choice_tokens = tokenize(choice)
        if choice_tokens.issubset(prediction_tokens):
            chosed_position = idx
            break
    
    return (cond1 and cond2), chosed_position

def mmsu_string_match(prediction, answer, choices, prompts):
    return letter_match(prediction, answer, choices, prompts)

def extract_letter_mapping_from_prompt(prompt):
    """
    Extracts a mapping from answer text to letter (A/B/C/D) from the prompt.
    Returns a dict: {answer_text: letter}
    Example:
        {"Hospital admissions": "A", ...}
    """
    # Remove instruction:
    prompt = prompt.lower()

    instruction_text = "(a) xxx.\n        do not add any other text.".lower()

    if instruction_text in prompt:
        prompt = prompt.split(instruction_text)[-1]

    instruction_text = "(a) xxx.\n    Do not add any other text.\n    \nuser\n".lower()

    if instruction_text in prompt:
        prompt = prompt.split(instruction_text)[-1]

    instruction_text = "Please answer only with the letter and the option value, e.g., '(A) Option value'.".lower()

    if instruction_text in prompt:
        prompt = prompt.split(instruction_text)[0]

    # Find lines like (A) Hospital admissions
    # Supports both upper and lower case letters
    # pattern = re.compile(r'\(\s*([A-Da-d])\s*\)\s*([^\n<]+)')
    pattern = re.compile(
        r'\(\s*([A-Za-z])\s*\)\s*(.*?)(?=\s*\(\s*[A-Za-z]\s*\)|<SEP>|Answer:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(prompt)
    mapping = {}
    for letter, text in matches:
        mapping[text.strip().rstrip(".")] = letter.upper()
    return mapping

def extract_letter_from_prediction(prediction):
    """
    Extracts letter from prediction string like '(C) Office building reception'
    Returns the letter as uppercase string, or None if not found.
    """
    m = re.search(r'[\s\.,;:-]*\(([A-Za-z])\)', str(prediction).strip())
    if m:
        return m.group(1).upper()
    return None

def letter_match(prediction, answer, choices, prompt)-> (bool, int):
    """
    Checks if the letter in the prediction matches the correct answer letter,
    using the mapping from the prompt.
    """
    
    # Build mapping from answer text to letter
    answer_to_letter = extract_letter_mapping_from_prompt(prompt)
    norm_answer = str(answer).strip().lower().rstrip(".")

    if "middle aged adult" == norm_answer:
        norm_answer = "middle-aged adult"

    # Find ground truth letter
    gt_letter = None
    found_match = False
    for ans_text, letter in answer_to_letter.items():
        normalized_text = ans_text.strip().lower().rstrip(".")
        
        if norm_answer == normalized_text:
            gt_letter = letter
            found_match = True
            break

    if not found_match:   
        gt_letter = "Z"  # Cover cases in which the answer is not within the choices

    # Extract predicted letter: check if prediction is already a letter
    if prediction and prediction[0] in "ABCDEFGHI":
        pred_letter = prediction[0]
    else:
        pred_letter = extract_letter_from_prediction(prediction)
    
    if not pred_letter:
        pred_letter = "Y"  # If not in expected format, ensure that they do not match

    chosed_position = ord(pred_letter) - ord('A')

    return pred_letter == gt_letter, chosed_position


if __name__ == "__main__":

    # Permutations
    permutations = ["answer_rp", "distractors_rp"]
    models = ["af2", "af3", "Qwen2.5-Omni-7B", "kimi-audio"]
    benchmarks = ["mmau-v05.15.25", "mmar", "mmsu"]

    longest_choice_results = []

    # Load the results
    results_folder = "results"

    gt_is_the_longest = []

    for model in models:
        #print(f"\n\nAnalyzing model: {model}")
        
        all_questions = []
        
        for permutation in permutations:
            
            for benchmark in benchmarks:
                # Load the results for each model and benchmark
                specific_results_folder = f"{results_folder}/{benchmark}/outputs/{permutation}_{model}/"
                #print(f"Processing {specific_results_folder}")

                if not os.path.exists(specific_results_folder):
                    print(f"Folder does not exist: {specific_results_folder}")
                    continue

                # List all the files in the folder
                all_files = os.listdir(specific_results_folder)
                all_json_files = [f for f in all_files if f.endswith(".json")]

                # Remove duplicated named files
                all_json_files = list(set(all_json_files))

                for json_file in all_json_files:
                    # Load the results
                    with open(os.path.join(specific_results_folder, json_file), "r") as f:
                        results_file = json.load(f)
                    #print(f"Loaded results from {json_file}, {len(results_file)} items")

                    # For every item, check if the model chose the longest choice
                    for item in results_file:
                        item_id = item["id"]
                        answer = item["answer"]
                        prediction = item["model_output"]
                        choices = item["choices"]
                        prompt = item["prompt"]

                        if "mmau" in benchmark:
                            is_correct, chosen_position = mmau_string_match(prediction, answer, choices, prompt)
                        elif "mmar" in benchmark:
                            try:
                                is_correct, chosen_position = mmau_string_match(prediction, answer, choices, prompt)
                            except:
                                #print(f"Error processing item ID {item_id} in benchmark {benchmark}. Skipping.")
                                continue
                        elif "mmsu" in benchmark:
                            is_correct, chosen_position = mmsu_string_match(prediction, answer, choices, prompt)
                        else:
                            raise ValueError(f"Unknown benchmark: {benchmark}")

                        # Calculate choice lengths in tokens
                        choice_lengths = [len(str(choice).split()) for choice in choices]
                        
                        # Find the index of the longest choice
                        longest_choice_idx = choice_lengths.index(max(choice_lengths))
                        
                        # Check if model chose the longest choice
                        chose_longest = (chosen_position == longest_choice_idx)
                        
                        # Check if the longest choice is correct
                        longest_is_correct = False
                        if chosen_position >= 0 and chosen_position < len(choices):
                            norm_answer = str(answer).strip().lower().rstrip(".")
                            if "middle aged adult" == norm_answer:
                                norm_answer = "middle-aged adult"
                            if "elderly adult" == norm_answer:
                                norm_answer = "elderly adult"
                            
                            longest_choice_text = str(choices[longest_choice_idx]).strip().lower().rstrip(".")
                            longest_is_correct = (longest_choice_text == norm_answer)

                        question_record = {
                            "question_id": item_id,
                            "benchmark": benchmark,
                            "permutation": permutation,
                            "chose_longest": chose_longest,
                            "longest_is_correct": longest_is_correct,
                            "choice_lengths": choice_lengths,
                            "max_length": max(choice_lengths),
                            "chosen_position": chosen_position,
                            "longest_choice_idx": longest_choice_idx,
                        }
                        all_questions.append(question_record)
        
        # Calculate metrics for this model
        df = pd.DataFrame(all_questions)
        
        if df.empty:
            print(f"No data available for model {model}. Skipping metrics calculation.")
            continue

        # print the dataframe head
        #print(df["chose_longest"].value_counts())
        # calculate overall percentage of choosing the longest choice
        overall_longest_choice_pct = df['chose_longest'].mean() * 100
        print(f"Model: {model}")
        print(f"Overall percentage of choosing the longest choice: {overall_longest_choice_pct:.2f}%")

        # Calculate percentage of choosing the longest choice when it is correct
        longest_correct_df = df[df['longest_is_correct']]
        if not longest_correct_df.empty:
            longest_choice_when_correct_pct = longest_correct_df['chose_longest'].mean() * 100
            print(f"Percentage of choosing the longest choice when it is correct: {longest_choice_when_correct_pct:.2f}%")


        # Calculate percentage of the longest choice being correct answer
        longest_is_correct_pct = df['longest_is_correct'].mean() * 100
        gt_is_the_longest.append(longest_is_correct_pct)
        print(f"Percentage of the longest choice being the correct answer: {longest_is_correct_pct:.2f}%")

    # Global percentage of the longest choice being correct answer
    if gt_is_the_longest:
        global_longest_is_correct_pct = np.mean(gt_is_the_longest)
        print(f"\nGlobal percentage of the longest choice being the correct answer across all models: {global_longest_is_correct_pct:.2f}%")