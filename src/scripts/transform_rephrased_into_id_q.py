import os
import json
import glob
import argparse


def main(args):
    
    # Read all rephrased files from the input folder
    rephrased_files = glob.glob(os.path.join(args.dataset_folder, "rephrased-q*speech*.json"))
    print(f"Found {len(rephrased_files)} rephrased question files.")
    print(f"Rephrased files: {rephrased_files}")

    if not rephrased_files:
        print("No rephrased question files found in the specified folder.")
        raise FileNotFoundError("No rephrased question files found.")

    # Read original file to copy original questions
    rephrasing_strategy = rephrased_files[0].split("/")[-1].split('_')[0]
    model_name = rephrased_files[0].split("/")[-1].split('_')[1]
    original_file_path = rephrased_files[0].replace(rephrasing_strategy + "_", "").replace(model_name + "_", "")
    with open(original_file_path, "r") as original_file:
        original_data = json.load(original_file)

    # get a dict with {id: question} from original data
    id_questions = {item["id"]: {"original_question": item["question"]} for item in original_data}


    for file_path in rephrased_files:
        print(f"Processing file: {file_path}")
        
        with open(file_path, "r") as infile:
            data = json.load(infile)

        rephrasing_strategy = file_path.split("/")[-1].split('_')[0]
        
        # For each id, add the rephrased question to the id_questions dict
        for item in data:
            question_id = item["id"]
            rephrased_question = item["question"]
            
            if question_id in id_questions:
                id_questions[question_id][rephrasing_strategy] = rephrased_question
            else:
                raise KeyError(f"ID {question_id} not found in original data.")

    
    # Transform the id_questions dict into a list of dictionaries
    id_questions = [
        {"id": q_id, **questions} for q_id, questions in id_questions.items()
    ]

    output_file = file_path.replace(".json", "_id_q.json")

    with open(output_file, "w") as outfile:
        json.dump(id_questions, outfile, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transform rephrased questions into ID questions.")
    parser.add_argument("--dataset_folder", type=str, help="Path to the dataset folder containing JSON files.")
    args = parser.parse_args()

    main(args)
