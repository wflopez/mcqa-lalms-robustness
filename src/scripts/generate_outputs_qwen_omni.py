# Copyright (c) 2025 Anonymous. 
#   Licensed under the MIT license.
# from safe_gpu import safe_gpu
# safe_gpu.claim_gpus()

import sys
sys.path.insert(1, sys.path[0].replace(sys.path[0].split('/')[-1], ''))

import os
import yaml
import json
import tqdm
import random
import argparse

import torch
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from utils.qwen_utils import process_mm_info

from safetensors.torch import load_file
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def predict(audio_path, prompt, sys_prompt):
    # print("prompt:", prompt)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True, zero_audio=parsed_args.zero_audio)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)
    output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=256, thinker_do_sample=False)
    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser("Generate outputs for Qwen/Qwen2.5-Omni-7B.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--output_filename", "-of", type=str, required=True, help="Output filename for the generated outputs")
    parser.add_argument("--audio_path", "-a", type=str, required=True, help="Path to audio files")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the choices in the input JSON")
    parser.add_argument("--zero_audio", action='store_true', help="Zero out the audio input")
    parser.add_argument("--reverse_options", action='store_true', help="Zero out the audio input")
    parser.add_argument("--extra_option", action='store_true', help="Zero out the audio input")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the Qwen2.5 Omni model")
    parser.add_argument('--cache_dir', type=str, default='models/', help='Directory to cache the model')
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

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        parsed_args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=parsed_args.cache_dir,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        parsed_args.model_name,
        cache_dir=parsed_args.cache_dir,
    )

    model = model.to(device)
    model.eval()

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

        if parsed_args.shuffle:
            random.shuffle(choices) # Shuffle the choices

        if parsed_args.reverse_options:
            # Reverse the order of the choices
            choices = choices[::-1]

        if len(choices) == 2:
            # Some question only have 2 choices
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}"
            if parsed_args.extra_option:
                text_prompt += "\n(C) None of the above"
        elif len(choices) == 3:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}"
            if parsed_args.extra_option:
                text_prompt += "\n(D) None of the above"
        elif len(choices) == 4:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}"
            if parsed_args.extra_option:
                text_prompt += "\n(E) None of the above"
        elif len(choices) == 5:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}"
            if parsed_args.extra_option:
                text_prompt += "\n(F) None of the above"
        elif len(choices) == 6:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}\n (F) {choices[5]}"
            if parsed_args.extra_option:
                text_prompt += "\n(G) None of the above"
        elif len(choices) == 8:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}\n (F) {choices[5]}\n (G) {choices[6]}\n (H) {choices[7]}"
            if parsed_args.extra_option:
                text_prompt += "\n(I) None of the above"
        else:
            raise ValueError(f"Unexpected number of choices: {len(choices)}")
        
        text_prompt += "\nPlease answer only with the letter and the option value, e.g., '(A) Option value'."        
        model_output = predict(audio_path, text_prompt, sys_prompt)

        # Parse the model output to get the answer
        if model_output and isinstance(model_output, list):
            model_output = model_output[0].strip()

        item["prompt"] = model_output.split("assistant")[0].strip()
        item["model_output"] = model_output.split("assistant")[-1].strip()
        model_outputs.append(item)

    # Save the model outputs to the output file
    output_file = os.path.join(parsed_args.output_folder, parsed_args.output_filename)
    with open(output_file, 'w') as f:
        json.dump(model_outputs, f, indent=4)

    print("*" * 30)
    print(f"Model outputs saved to {output_file}")
