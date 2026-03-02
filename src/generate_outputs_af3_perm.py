# Copyright (c) 2025 Anonymous. 
#   Licensed under the MIT license.

import os
import sys
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
from pydub import AudioSegment
from safetensors.torch import load_file

from audio_flamingo_3.inference_utils import load_model, predict


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser("Generate outputs for Audio Flamingo 3.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--output_filename", "-of", type=str, required=True, help="Output filename for the generated outputs")
    parser.add_argument("--audio_path", "-a", type=str, required=True, help="Path to audio files")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml configuration file")
    parser.add_argument("--perm_idx", type=int, default=0, choices=range(0,24), metavar="[0-23]", help="Permutation index for the choices in the input JSON")
    parsed_args = parser.parse_args()

    os.makedirs(parsed_args.output_folder, exist_ok=True)

    # Check if output file already exists
    output_file = os.path.join(parsed_args.output_folder, parsed_args.output_filename)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Please remove it or choose a different output filename.")
        sys.exit(1)

    # Print the arguments
    print("Running arguments:", vars(parsed_args))
    print("*" * 30)

    # Set seed for reproducibility
    random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    # Read the configuration file
    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, generation_config = load_model(
        device=device,
        config=config
    )
    model.eval()

    # Read the reference data json:
    with open(parsed_args.input, 'r') as f:
        data = json.load(f)

    model_outputs = []
        
    # Iterate over the reference data
    for item in tqdm.tqdm(data):
        if "audio_path" in item:
            audio_path = item["audio_path"]
            audio_path = os.path.join(parsed_args.audio_path, audio_path.split("/")[-1])
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

        model_output = predict((model, generation_config), audio_path, question)

        item["prompt"] = question
        item["model_output"] = model_output
        model_outputs.append(item)

    # Save the model outputs to the output file
    with open(output_file, 'w') as f:
        json.dump(model_outputs, f, indent=4)

    print("*" * 30)
    print(f"Model outputs saved to {output_file} using permutation index {parsed_args.perm_idx}.\n\n")
