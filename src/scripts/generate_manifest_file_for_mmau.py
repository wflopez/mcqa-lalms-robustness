import os
import json
import random
import argparse

import torchaudio


def transform_data(input_list, split_path, split, flamingo_task):
    transformed = {
        "split": split,
        "split_path": split_path,
        "flamingo_task": flamingo_task,
        "total_num": len(input_list),
        "data": {}
    }

    for idx, item in enumerate(input_list):
        transformed["data"][str(idx)] = {
            "name": os.path.basename(item["audio_id"]),
            "prompt": item["question"],
            "output": item["answer"],
            "duration": item["duration"],
        }

    return transformed


def main(args):

    # Set the random seed for reproducibility
    random.seed(args.seed)

    dataset_file = args.dataset_json_file

    # Get dataset name from the file name
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]

    output_folder = args.output_folder

    # Read the input file
    with open(dataset_file, "r") as f:
        data = json.load(f)

    data_with_durations = []

    for item in data:
        try:
            audio_id = item["audio_id"]
        except KeyError:
            audio_id = item["audio_path"]
        # Get only the audio file name
        audio_id = os.path.basename(audio_id)
        audio_path = os.path.join(args.audio_folder, audio_id)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} does not exist.")
            
        # Torch info
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        item["duration"] = duration

        # Include the options in the prompt with the question
        # Following the format -> Question? (A) xxx. (B) yyy. (C) zzz. (D) uuu.
        question = item["question"]
        choices = item["choices"]

        if args.shuffle:
            random.shuffle(choices) # Shuffle the choices

        # Question. Choose the correct option from the following options:\n(A) xxx\n(B) yyy\n(C) zzz\n(D) uuu

        if len(choices) == 2:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}"
        if len(choices) == 3:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}"
        if len(choices) == 4:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
        if len(choices) == 5:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n(E) {choices[4]}"
        if len(choices) == 6:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n(E) {choices[4]}\n(F) {choices[5]}"
        if len(choices) == 7:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n(E) {choices[4]}\n(F) {choices[5]}\n(G) {choices[6]}"
        if len(choices) == 8:
            item["question"] = f"{question} Choose the correct option from the following options:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n(E) {choices[4]}\n(F) {choices[5]}\n(G) {choices[6]}\n(H) {choices[7]}"
        else:
            raise ValueError(f"Unsupported number of choices: {len(choices)}. Expected 2 to 8 choices.")
            
        item["audio_id"] = audio_id

        data_with_durations.append(item)
    
    # Transform the data
    transformed_data = transform_data(
        data_with_durations,
        split_path = "MMAU/test",
        split="test",
        flamingo_task="MMAU",
        )

    # Save the transformed data to a JSON file
    output_file = os.path.join(output_folder, dataset_name + "_manifest.json")
    with open(output_file, "w") as f:
        json.dump(transformed_data, f, indent=4)
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate manifest file for MMAU")

    parser.add_argument("--dataset_json_file", type=str, required=True, help="Path to the dataset file root")
    parser.add_argument("--audio_folder", type=str, required=True, help="Path to the folder containing audio samples")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the choices in the input JSON")

    args = parser.parse_args()

    main(args)




