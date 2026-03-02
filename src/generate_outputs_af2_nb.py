# Copyright (c) 2025 Anonymous. 
#   Licensed under the MIT license.

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
from safetensors.torch import load_file

from audio_flamingo_2.factory import create_model_and_transforms
from audio_flamingo_2.utils import Dict2Class, get_autocast, get_cast_dtype, int16_to_float32, float32_to_int16


def get_num_windows(T, sr, clap_config):

    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    
    return num_windows, full_length


def read_audio(file_path, target_sr, duration, start, clap_config):

    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    
    assert len(data.shape) == 1, data.shape
    return data


def load_audio(audio_path, clap_config):

    sr = 16000
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

    audio_data = read_audio(audio_path, sr, duration, 0.0, clap_config) # hard code audio start to 0.0
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    # pads to the nearest multiple of window_length
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) > max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)
    
    return audio_clips, audio_embed_mask


def predict(filepath, question, clap_config, inference_kwargs):

    audio_clips, audio_embed_mask = load_audio(filepath, clap_config)
    audio_clips = audio_clips.to(device, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device, dtype=cast_dtype, non_blocking=True)

    text_prompt = str(question).lower()

    sample = f"<audio>{text_prompt.strip()}{tokenizer.sep_token}"

    text = tokenizer(
        sample,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device, non_blocking=True)

    prompt = input_ids

    with torch.no_grad():
        output = model.generate(
            audio_x=audio_clips.unsqueeze(0),
            audio_x_mask=audio_embed_mask.unsqueeze(0),
            lang_x=prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            **inference_kwargs,
            temperature=0.0
        )[0]
    
    output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')

    #print('Prompt: ', question)
    #print('Audio Flamingo 2: ', output_decoded)

    return output_decoded


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser("Generate outputs for Audio Flamingo 2 on MMAU test set.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--output_filename", "-of", type=str, required=True, help="Output filename for the generated outputs")
    parser.add_argument("--audio_path", "-a", type=str, required=True, help="Path to audio files")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml configuration file")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the choices in the input JSON")
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

    # Read the configuration file
    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    inference_config = config['inference_config']
    args = Dict2Class(config['train_config'])

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        device=device
    )

    model = model.to(device)
    model.eval()

    # Load the pretrained weights: XATTN and Transformer
    ckpt_path = inference_config['pretrained_path']
    metadata_path = os.path.join(ckpt_path, "safe_ckpt/metadata.json")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct the full state_dict
    state_dict = {}

    # Load each SafeTensors chunk
    for chunk_name in metadata:
        chunk_path = f"safe_ckpt/{chunk_name}.safetensors"
        chunk_tensors = load_file(os.path.join(ckpt_path, chunk_path))

        # Merge tensors into state_dict
        state_dict.update(chunk_tensors)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    print("Model loaded successfully.")
    print("*" * 30)

    # Cast the model to the appropriate dtype
    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )
    cast_dtype = get_cast_dtype(args.precision)

    # Use greedy decoding for inference
    inference_kwargs = {
        "do_sample": False,
        "top_k": 30,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    # Read the reference data json:
    with open(parsed_args.input, 'r') as f:
        data = json.load(f)

    model_outputs = []
        
    # Iterate over the refernce data
    for item in tqdm.tqdm(data):
        if "audio_path" in item:
            audio_path = item["audio_path"]
            audio_path = os.path.join(parsed_args.audio_path, audio_path.split("/")[-1])
        else:
            audio_path = os.path.join(parsed_args.audio_path, item["id"]+".wav")
        question = item["question"]
        choices = item["choices"]. copy()  # Make a copy of the choices to avoid modifying the original data

        if parsed_args.shuffle:
            #print("Shuffling the choices, original order:", choices)
            random.shuffle(choices) # Shuffle the choices
            #print("Shuffling the choices, new order:", choices)

        #NOTE: MMAU includes different number of choices
        if len(choices) == 2:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}"
        elif len(choices) == 3:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}"
        elif len(choices) == 4:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}"
        elif len(choices) == 5:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}"
        elif len(choices) == 6:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}\n (F) {choices[5]}"
        elif len(choices) == 7:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}\n (F) {choices[5]}\n (G) {choices[6]}"
        elif len(choices) == 8:
            text_prompt = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n (C) {choices[2]}\n (D) {choices[3]}\n (E) {choices[4]}\n (F) {choices[5]}\n (G) {choices[6]}\n (H) {choices[7]}"
        else:
            raise ValueError(f"Unexpected number of choices: {len(choices)}")

        #print("Question:", question)
        model_output = predict(audio_path, text_prompt, clap_config, inference_kwargs)

        item["prompt"] = text_prompt
        item["model_output"] = model_output
        model_outputs.append(item)

    # Save the model outputs to the output file
    output_file = os.path.join(parsed_args.output_folder, parsed_args.output_filename)
    with open(output_file, 'w') as f:
        json.dump(model_outputs, f, indent=4)

    print("*" * 30)
    print(f"Model outputs saved to {output_file}")
