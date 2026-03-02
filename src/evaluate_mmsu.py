import argparse
import json
from tqdm import tqdm
from pathlib import Path
import os
import re

global count_answer_not_present
count_answer_not_present = 0

def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', str(text).lower()))
    
    # Tokenize prediction and answer
    try:
        prediction_tokens = tokenize(prediction)
    except Exception as e:
        print(f"Error tokenizing prediction: {e}")
        return False
    try:
        answer_tokens = tokenize(answer)
    except Exception as e:
        print(f"Error tokenizing answer: {e}")
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    try:
        for choice in choices:
            choice_tokens = tokenize(choice)
            if choice_tokens != answer_tokens:
                incorrect_tokens.update(choice_tokens - answer_tokens)
    except Exception as e:
        print(f"Error processing choices: {e}")
        return False
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2


def exact_match(answer, prediction, choices, prompt):
    # Remove choice letter (a), (b), etc. from the prediction
    prediction = re.sub(r'^\s*\([a-z]\)\s*', '', str(prediction), flags=re.IGNORECASE).strip()
    
    # Normalize the answer
    answer = str(answer).strip().lower()
    
    # Normalize the prediction
    prediction = str(prediction).strip().lower()
    
    return answer == prediction


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
    #pattern = re.compile(r'\(\s*([A-Da-d])\s*\)\s*([^\n<]+)')
    pattern = re.compile(
        r'\(\s*([A-Za-z])\s*\)\s*(.*?)(?=\s*\(\s*[A-Za-z]\s*\)|<SEP>|Answer:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(prompt)
    mapping = {}
    for letter, text in matches:
        #print(f"Mapping: {letter} -> {text.strip()}")
        mapping[text.strip().rstrip(".")] = letter.upper()
    return mapping

def extract_letter_from_prediction(prediction):
    """
    Extracts letter from prediction string like '(C) Office building reception'
    Returns the letter as uppercase string, or None if not found.
    """
    #m = re.match(r'[\s\.,;:-]*\(([A-Da-d])\)', str(prediction))
    m = re.search(r'[\s\.,;:-]*\(([A-Za-z])\)', str(prediction).strip())
    if m:
        return m.group(1).upper()
    return None

def letter_match(answer, prediction, choices, prompt):
    """
    Checks if the letter in the prediction matches the correct answer letter,
    using the mapping from the prompt.

    Args:
        answer: The ground truth answer text (e.g., "Office building reception").
        prediction: The model output (e.g., "(C) Office building reception").
        choices: List of answer texts (order may not match A/B/C/D).
        prompt: The prompt containing the letter mappings.

    Returns:
        True if predicted letter matches ground truth letter, False otherwise.
    """
    # Build mapping from answer text to letter
    answer_to_letter = extract_letter_mapping_from_prompt(prompt)
    # print("answer to letter", answer_to_letter)

    # Normalize answer for matching
    norm_answer = str(answer).strip().lower().rstrip(".")

    if "middle aged adult" == norm_answer:
        norm_answer = "middle-aged adult"

    # Find ground truth letter
    gt_letter = None
    for ans_text, letter in answer_to_letter.items():
        #print(f"Comparing '{norm_answer}' with '{ans_text.strip().lower().rstrip('.')}'")
        if norm_answer == ans_text.strip().lower().rstrip("."):
            gt_letter = letter
            break

    if not gt_letter:
        #print(f"Ground truth answer '{answer}' not found in prompt mapping.")
        #print(f"Prompt was: {prompt}")
        #print(f"Answer to letter:", answer_to_letter)
        gt_letter = "Z"  # Cover cases in which the answer is not within the choices
        global count_answer_not_present
        count_answer_not_present += 1
        #print(f"Ground truth answer '{answer}' not found in prompt mapping {answer_to_letter}")
        #print(f"Prompt: {prompt}")

    # print(f"Ground truth letter: {gt_letter}")

    # Extract predicted letter
    # Check if prediction is already a letter
    if prediction and prediction[0] in "ABCDEFGHI":
        pred_letter = prediction[0]
    else:
        pred_letter = extract_letter_from_prediction(prediction)
    
    if not pred_letter:
        #print(f"Prediction '{prediction}' is not in the expected format.")
        pred_letter = "Y" # If not in expected format, ensure that they do not match

    # print(f"Predicted letter: {pred_letter}")

    return pred_letter == gt_letter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance with nested categories")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--dest_folder', type=str, required=True, help='Output folder for results')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        input_data = json.load(f)

    # Initialize metrics dictionaries
    task_metrics = {}
    category_metrics = {}
    subcat_metrics = {}
    subsubcat_metrics = {}
    
    corr, total, no_pred_count = 0, 0, 0
    #if "kimi" in args.input.lower():
    #    output_key = 'original_output'
    #else:
     #   output_key = 'model_output'
    output_key = 'model_output'
    
    new_data = []

    for idx, sample in enumerate(tqdm(input_data)):
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        choices = sample['choices']
        _prompt = sample['prompt']
        
        # Extract category values
        task_val = sample.get('task')
        category_val = sample.get('category')
        subcat_val = sample.get('sub-category')
        subsubcat_val = sample.get('sub-sub-category')
        
        # Initialize category metrics
        for val, metric_dict in [
            (task_val, task_metrics),
            (category_val, category_metrics),
            (subcat_val, subcat_metrics),
            (subsubcat_val, subsubcat_metrics)
        ]:
            if val and val not in metric_dict:
                metric_dict[val] = [0, 0]  # [correct, total]

        #match_result = string_match(_answer, _prediction, choices)
        #match_result = exact_match(_answer, _prediction, choices, _prompt)
        match_result = letter_match(_answer, _prediction, choices, _prompt)

        if match_result:
            corr += 1
            sample['match'] = 1
            # Update correct and total counts for all categories
            for val, metric_dict in [
                (task_val, task_metrics),
                (category_val, category_metrics),
                (subcat_val, subcat_metrics),
                (subsubcat_val, subsubcat_metrics)
            ]:
                if val:
                    metric_dict[val][0] += 1
                    metric_dict[val][1] += 1
        else:
            sample['match'] = 0
            # Update only total counts for categories
            for val, metric_dict in [
                (task_val, task_metrics),
                (category_val, category_metrics),
                (subcat_val, subcat_metrics),
                (subsubcat_val, subsubcat_metrics)
            ]:
                if val:
                    metric_dict[val][1] += 1

            #print(f"[DEBUG] Sample {idx} - No match: Answer: {sample['answer']}, Prediction: {_prediction}, Choices: {choices}")

        total += 1
        new_data.append(sample)

    # Compile results
    results = {
        'task': {k: (v[0]/v[1])*100 for k, v in task_metrics.items()},
        'category': {k: (v[0]/v[1])*100 for k, v in category_metrics.items()},
        'sub-category': {k: (v[0]/v[1])*100 for k, v in subcat_metrics.items()},
        'sub-sub-category': {k: (v[0]/v[1])*100 for k, v in subsubcat_metrics.items()},
        'total': (corr/total)*100,
        'no_prediction': no_pred_count
    }

    print(results)

    # Print results
    for category_type in ['task', 'category', 'sub-category', 'sub-sub-category']:
        print(f"\n{category_type.replace('-', ' ').title()} Accuracy:")
        for name, acc in results[category_type].items():
            count = locals()[f"{category_type.replace('-', '_')}_metrics".replace('sub_sub_category', 'subsubcat').replace('sub_category', 'subcat')][name][1]
            print(f"{name}: {acc:.2f}% ({count} samples)")

    print(f"\nTotal Accuracy: {results['total']:.2f}%")
    print(f"No Prediction Count: {no_pred_count}")

    # Save results
    output_file = os.path.basename(args.input).replace('.json', '_results.json')
    output_path = os.path.join(args.dest_folder, output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")
    # print in red
    if count_answer_not_present == 4:
        print(f"Count of samples without ground truth answer: {count_answer_not_present}")
    else:
        print(f"\033[91mCount of samples without ground truth answer: {count_answer_not_present}\033[0m")
