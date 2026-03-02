import os
import json
import random
import argparse


def main(args):

    # Load the reference json file
    with open(args.reference_file, 'r') as f:
        reference_data = json.load(f)

    trivial_results = []

    for sample in reference_data:
        # read choises
        choices = sample.get('choices', [])
        
        # Check if the trivial index is valid
        if args.trivial_idx < 0:
            raise ValueError("Trivial index must be a non-negative integer.")
        elif args.trivial_idx >= len(choices):
            print(f"Warning: Trivial index {args.trivial_idx} is out of range for choices. Using the random choice instead.")
            # Randomly select a choice if the index is out of range
            trivial_choice = random.choice(choices)
        else:
            # Use the trivial choice based on the provided index
            trivial_choice = choices[args.trivial_idx]

        # Add a "model_output" field to the sample
        sample['model_output'] = trivial_choice

        # Add the sample to the random results
        trivial_results.append(sample)

    print("Fake results generated successfully.")
    
    # Print the first random result for verification
    print("First random result:", trivial_results[0])

    # Save into a new json file
    output_file = args.reference_file.split('/')[-1].replace('.json', '_trivial-' + str(args.trivial_idx)  + '.json')
    results_path = os.path.join(args.dest_folder, output_file)

    print(f"Saving results to {results_path}")

    with open(results_path, 'w') as f:
        json.dump(trivial_results, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate fake results for evaluation.")
    parser.add_argument('--reference_file', type=str, required=True, help='Path to input JSON with evaluation data')
    parser.add_argument('--trivial_idx', type=int, default=0, help='Index of the trivial choice to use as model output')
    parser.add_argument('--dest_folder', type=str, required=True, help='Path to destination folder to save the random results')
    args = parser.parse_args()
    
    main(args)