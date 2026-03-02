import os
import json
import random
import argparse


def main(args):

    # Set the random seed for reproducibility
    random.seed(2025)

    # Load the reference json file
    with open(args.reference_file, 'r') as f:
        reference_data = json.load(f)

    random_results = []

    for sample in reference_data:
        # read choises
        choices = sample.get('choices', [])
        
        # Ramdon sample one from the choices
        random_choice = random.choice(choices)

        # Add a "model_output" field to the sample
        sample['model_output'] = random_choice

        # Add the sample to the random results
        random_results.append(sample)

    print("Fake results generated successfully.")
    
    # Print the first random result for verification
    print("First random result:", random_results[0])

    # Save into a new json file
    output_file = args.reference_file.split('/')[-1].replace('.json', '_random.json')
    results_path = os.path.join(args.dest_folder, output_file)

    print(f"Saving results to {results_path}")

    with open(results_path, 'w') as f:
        json.dump(random_results, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate fake results for evaluation.")
    parser.add_argument('--reference_file', type=str, required=True, help='Path to input JSON with evaluation data')
    parser.add_argument('--dest_folder', type=str, required=True, help='Path to destination folder to save the random results')
    args = parser.parse_args()
    
    main(args)