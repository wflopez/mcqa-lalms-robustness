import sys
sys.path.insert(1, sys.path[0].replace(sys.path[0].split('/')[-1], ''))

import os
import re
import yaml
import json
import tqdm
import random
import argparse
import itertools

import torch
import librosa
import numpy as np
import soundfile as sf

from kimia_infer.api.kimia import KimiAudio


def extract_option_mapping(question):
    """
    Parses question text to extract mapping from option letters to answer strings.
    Returns a dict, e.g. {'A': 'apple', 'B': 'banana', ...}
    """
    # Regex to match lines like (A) apple or (B) banana
    pattern = re.compile(r'\(([A-Z])\)\s*([^\n]+)')
    mapping = {}
    for match in pattern.finditer(question):
        letter = match.group(1).upper()
        answer = match.group(2).strip()
        mapping[letter] = answer
    return mapping

def get_output_from_letter(output, mapping):
    """
    Extracts the answer text from a letter in the output.

    "What is the mood of this audio?\n(A) Happy\n(B) Sad\n(C) Angry\n(D) Relaxed"
    Model returns: A
    This method returns: (A) Happy
    """
    # Extract the letter from the output
    letter = output.strip().upper()
    # Return the corresponding answer text from the mapping
    return mapping.get(letter, "")

def predict(audio_path, prompt, sys_prompt):
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    # --- 3. Audio QA ---
    messages_audio_qa = [
        # You can provide context or instructions as text
        {"role": "user", "message_type": "text", "content": sys_prompt},
        # Provide the question as text
        {"role": "user", "message_type": "text", "content": prompt},
        # Provide the audio file path
        {"role": "user", "message_type": "audio", "content": audio_path}
    ]

    _, original_output = model.generate(messages_audio_qa, **sampling_params, output_type="text")

    # Check if text output contains only a letter, like: "A" then apply the mapping
    # And generate "(A) xxx"
    if original_output and len(original_output.strip()) == 1 and original_output.strip().isalpha():
        mapping = extract_option_mapping(prompt)
        text_output = get_output_from_letter(original_output, mapping)
        text_output = f"({original_output.strip().upper()}) {text_output}"
    else:
        # If the output is not a single letter, return it as is
        text_output = original_output
        
    return text_output.strip(), original_output.strip()

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser("Generate outputs for Qwen/Qwen2.5-Omni-7B.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--output_filename", "-of", type=str, required=True, help="Output filename for the generated outputs")
    parser.add_argument("--audio_path", "-a", type=str, required=True, help="Path to audio files")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--zero_audio", action='store_true', help="Zero out the audio input")
    parser.add_argument("--extra_option", action='store_true', help="Zero out the audio input")
    parser.add_argument('--cache_dir', type=str, default='models/', help='Directory to cache the model')
    parser.add_argument("--model_name", type=str, default="kimi-audio", help="Name of the Kimi-Audio model to use")
    parser.add_argument("--perm_idx", type=int, default=0, choices=range(0,24), metavar="[0-23]", help="Permutation index for the choices in the input JSON")
    parsed_args = parser.parse_args()

    print("Running arguments:", vars(parsed_args))
    print("*" * 30)

    os.makedirs(parsed_args.output_folder, exist_ok=True)

    # Check if output file already exists
    output_file = os.path.join(parsed_args.output_folder, parsed_args.output_filename)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or choose a different output filename.")
        sys.exit(1)

    # Set seed for reproducibility
    random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Kimi-Audio model
    model_path = os.path.join(parsed_args.cache_dir, parsed_args.model_name)
    model = KimiAudio(model_path=model_path, load_detokenizer=False)

    # Read the reference data json:
    with open(parsed_args.input, 'r') as f:
        data = json.load(f)

    sys_prompt = """Answer the user's question without providing extra information, choice one of the options.
    Choices are in the format: (a) xxx. (b) yyy. (c) zzz. (d) uuu.
    Return only one of the options in the format: (a) xxx.
    Do not add any other text.
    """
    model_outputs = []
        
    # Iterate over the refernce data
    for item in tqdm.tqdm(data):
        # Look for "audio_path" in the item
        if "audio_path" in item:
            audio_path = item["audio_path"]
            audio_path = os.path.join(parsed_args.audio_path, audio_path.split("/")[-1])  # Ensure the audio path is correct
        else:
            audio_path = os.path.join(parsed_args.audio_path, item["id"]+".wav")

        question = item["question"]
        choices = item["choices"]. copy()  # Make a copy of the choices to avoid modifying the original data
        answer = item["answer"]

        # When there are more that 4 choices: truncate the choices to 4 including the correct answer
        if len(choices) > 4:
            # Remove correct answer from the choices list
            if answer in choices: # Some questions do not have the answer in the choices in MMAR!!!!!
                choices.remove(answer)
            # Randomly sample 3 choices from the remaining choices and add the correct answer
            choices = [answer] + random.sample(choices, 3)
        
        # Calulate all possible permutations of the choices
        permutations = list(itertools.permutations(choices))
        
        # Select the permutation based on the perm_idx argument
        if len(permutations) == 2:
            # Some question only have 2 choices -> 2 possible permutations
            # For index bigger than 1, we will take the second permutation
            idx_to_use = parsed_args.perm_idx
            if (parsed_args.perm_idx > 1):
                idx_to_use = -1
            choices = list(permutations[idx_to_use])
            question = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}"
        elif len(permutations) == 6:
            # Some question only have 3 choices -> 6 possible permutations
            # For index bigger than 6, we will take the third possible permutation
            idx_to_use = parsed_args.perm_idx
            if (parsed_args.perm_idx > 5):
                idx_to_use = -1
            choices = list(permutations[idx_to_use])
            question = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}"
        elif len(permutations) == 24:
            choices = list(permutations[parsed_args.perm_idx])
            question = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
        else:
            raise ValueError(f"Unexpected number of permutations: {len(permutations)}. Expected 24 permutations for 4 choices.")
        
        #question += "\nPlease answer only with the letter and the option value, e.g., '(A) Option value'."        
        model_output, original_output = predict(audio_path, question, sys_prompt)

        # Parse the model output to get the answer
        if model_output and isinstance(model_output, list):
            model_output = model_output[0].strip()

        item["prompt"] = question
        item["model_output"] = model_output.split("assistant")[-1].strip()
        item["original_output"] = original_output.strip()
        model_outputs.append(item)

    # Save the model outputs to the output file
    output_file = os.path.join(parsed_args.output_folder, parsed_args.output_filename)
    with open(output_file, 'w') as f:
        json.dump(model_outputs, f, indent=4)

    print("*" * 30)
    print(f"Model outputs saved to {output_file}")
