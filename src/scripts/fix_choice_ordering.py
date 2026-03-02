import json
import difflib


def fix_choice_ordering(reference_file, input_file, output_file):
    with open(reference_file, "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Get position of the gt on the reference data
    reference_map = {}
    for item in reference_data:
        if item["answer"] in item["choices"]:
            reference_map[item["id"]] = item["choices"].index(item["answer"])
        elif "mmsu" in reference_file.lower():
            # Try to find the closest match using difflib
            if "middle aged adult" == item["answer"]:
                reference_map[item["id"]] = item["choices"].index("Middle-aged adult")
            elif "elderly adult" == item["answer"]:
                reference_map[item["id"]] = item["choices"].index("Elderly adult")
            else:
                print(f"Warning: answer '{item['answer']}' not found in choices for id '{item['id']}' and no close match found")
                reference_map[item["id"]] = 0
        else:
            print(f"Warning: answer '{item['answer']}' not found in choices for id '{item['id']}'")
            # Put gt in the first position if not found
            reference_map[item["id"]] = 0

    fixed_data = []
    for item in input_data:
        ref_position = reference_map.get(item["id"])
        old_index = item["choices"].index(item["answer"])
        item["choices"].insert(ref_position, item["choices"].pop(old_index))
        fixed_data.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    #datasets = ["MMAU-v05.15.25", "MMAR", "MMSU"]
    datasets = ["MMSU"]
    rephrasings = [
        "rephrased-a-with-d_gemma-3-12b-it", # NOTE: THIS IS STILL REQUIRED
        #"rephrased-a-with-q_gemma-3-12b-it",
        #"rephrased-a-with-qd_gemma-3-12b-it",
        ]

    for dataset in datasets:
        if dataset == "MMAU-v05.15.25":
            print("Using MMAU dataset")
            dataset_path="/mnt/matylda4/xlopezw00/MMAU-v05.15.25/"
            dataset_filename="mmau-test-mini.json"
        elif dataset == "MMAR":
            print("Using MMAR dataset")
            dataset_path="/mnt/matylda4/xlopezw00/MMAR/"
            dataset_filename="MMAR-meta.json"
        elif dataset == "MMSU":
            print("Using MMSU dataset")
            dataset_path="/mnt/matylda4/xlopezw00/MMSU/question/"
            dataset_filename="mmsu.json"
        else:
            print(f"Unknown dataset: {dataset}")
            raise ValueError(f"Unknown dataset: {dataset}")

        for rephrasing in rephrasings:
            print(f"Processing rephrasing: {rephrasing}")
            reference_file = f"{dataset_path}{dataset_filename}"
            input_file = f"{dataset_path}{rephrasing}_{dataset_filename}"
            output_file = f"{dataset_path}rephrase_a_fixed/{rephrasing}_{dataset_filename}"
            fix_choice_ordering(reference_file, input_file, output_file)

            print(f"Reference file: {reference_file}")
            print(f"Input file: {input_file}")
            print(f"Fixed file saved to: {output_file}")


    