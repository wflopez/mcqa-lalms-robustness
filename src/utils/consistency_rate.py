import re
from itertools import combinations
from typing import List


def calculate_consistency_rate(responses: List[List[str]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    total_similarity = 0
    total_combinations = 0

    for response_set in responses:
        pairs = combinations(response_set, 2)
        num_pairs = len(response_set) * (len(response_set) - 1) / 2
        total_combinations += num_pairs
        for answer1, answer2 in pairs:
            total_similarity += int(answer1 == answer2)

    return total_similarity / total_combinations if total_combinations > 0 else 0.0


def mmau_string_match(prediction, answer, choices, prompts):
    """String matching for MMAU and MMAR benchmarks."""
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
    
    return cond1 and cond2


def extract_letter_mapping_from_prompt(prompt):
    """
    Extracts a mapping from answer text to letter (A/B/C/D) from the prompt.
    Returns a dict: {answer_text: letter}
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


def letter_match(prediction, answer, choices, prompt):
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
    for ans_text, letter in answer_to_letter.items():
        normalized_text = ans_text.strip().lower().rstrip(".")
        if norm_answer == normalized_text:
            gt_letter = letter
            break

    if not gt_letter:
        gt_letter = "Z"  # Cover cases in which the answer is not within the choices

    # Extract predicted letter
    # Check if prediction is already a letter
    if prediction and prediction[0] in "ABCDEFGHI":
        pred_letter = prediction[0]
    else:
        pred_letter = extract_letter_from_prediction(prediction)
    
    if not pred_letter:
        pred_letter = "Y"  # If not in expected format, ensure that they do not match

    return pred_letter == gt_letter


def mmsu_string_match(answer, prediction, choices, prompts):
    """String matching for MMSU benchmark."""
    return letter_match(answer, prediction, choices, prompts)


def calculate_consistent_and_correct_rate(responses: List[List[str]], gt_answers: List[List[str]], choices_list: List[List[List[str]]], prompts_list: List[List[str]], benchmark: str) -> float:
    """
    Calculate the Consistency and Correct Rate (CCR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.
    gt_answers: List of ground truth answers corresponding to each response set.
    choices_list: List of choice lists corresponding to each response set.
    prompts_list: List of prompts corresponding to each response set.
    benchmark: Benchmark name ("mmar", "mmau", or "mmsu")

    Returns:
    The consistency rate as a float.
    """
    if (benchmark == "mmar") or (benchmark == "mmau"):
        correctness = mmau_string_match
    else:
        correctness = mmsu_string_match

    consistent_and_correct = 0

    for response_set, gt_set, choices_set, prompt_set in zip(responses, gt_answers, choices_list, prompts_list):
        if all(correctness(response, gt, choices, prompt) for response, gt, choices, prompt in zip(response_set, gt_set, choices_set, prompt_set)):
            consistent_and_correct += 1
    return consistent_and_correct / len(responses)
