import os
import re
import json
import argparse
import pandas as pd


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
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


def main(args):

    models_list = [
        'gemma-3-4b-it',
        'gemma-3-12b-it',
        'gemma-3-27b-it',
        'Llama-3.1-8B-Instruct',
        'Qwen2.5-7B-Instruct',
        #'Qwen2.5-3B-Instruct',
    ]

    ids_with_diff_answers = [
        "BV1GH4y1p7vE_00-01-00_00-01-26",
        "QtIlD6uxwwk_00-00-33_00-00-40",
        "pRgr18CWiFE_00-00-00_00-00-24",
        "1haxFVCxSJI_00-00-00_00-00-05",
        "1haxFVCxSJI_00-00-00_00-00-03",
        "Z8nnssMvvG4_00-00-00_00-00-07",
        "BV1zso1YkENT_00-00-00_00-00-14",
        "zar3EpCxKr0_00-01-05_00-01-27",
        "bbabe360-0573-43d4-b2e6-6892150cbdcd",
        "42e35733-9630-4960-9ce0-75325e5906f6",
    ]

    # Look for files in args.outputs_path that match the models in models_list
    files = []
    if "mmar" in args.outputs_path:
        files += [f"MMAR-meta_{model}.json" for model in models_list]
        benchmark_name = "mmar"
    if "mmau" in args.outputs_path:
        files += [f"mmau-test-mini_{model}.json" for model in models_list]
        benchmark_name = "mmau"
    if "mmsu" in args.outputs_path:
        files += [f"mmsu_{model}.json" for model in models_list]
        benchmark_name = "mmsu"

    list_of_ids = []

    # Read json files and filter questions answered by LLMs
    for model_file in files:
        try:
            with open(f"{args.outputs_path}/{model_file}", 'r') as file:
                data = json.load(file)

            for item in data:
                # Check if the model has answered the question correctly
                model_output = item.get('model_output', '')
                answer = item.get('answer', '')
                
                # Normalize the model output and answer
                model_output = model_output \
                .replace("(A) ", "") \
                .replace("(B) ", "") \
                .replace("(C) ", "") \
                .replace("(D) ", "") \
                .replace("(E) ", "") \
                .replace("(F) ", "") \
                .replace("(G) ", "") \
                .replace("(H) ", "") \
                .replace("(I) ", "")
                model_output = model_output \
                .replace("(a) ", "") \
                .replace("(b) ", "") \
                .replace("(c) ", "") \
                .replace("(d) ", "") \
                .replace("(e) ", "") \
                .replace("(f) ", "") \
                .replace("(g) ", "") \
                .replace("(h) ", "") \
                .replace("(i) ", "")

                # Remove puntuation and convert to lowercase ". , ; : ! ?"
                model_output = model_output.translate(str.maketrans('', '', '.,;:!?'))
                model_output = model_output.lower().strip()
                answer = answer.translate(str.maketrans('', '', '.,;:!?'))
                answer = answer.lower().strip()
                
                sample_id = item['id']
                
                if sample_id in ids_with_diff_answers:
                    print(f"Skipping sample {sample_id} due to multiple answers.")
                    continue
                
                """
                # Check if the model output matches the answer
                if model_output == answer:
                    list_of_ids.append([item['id'], item['question'], item['answer'], model_output, model_file.split('_')[1].replace('.json', '')])
                """

                if string_match(answer, model_output, item.get('choices', [])):
                    print(f"Model output: {model_output}, Answer: {answer}")
                    list_of_ids.append([sample_id, item['question'], item['answer'], item['choices'], model_output, model_file.split('_')[1].replace('.json', '')])
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File {model_file} not found in {args.outputs_path}. Please check the path and file names.")

    # Build a dataframe with the questions answered correctly by the LLMs
    df = pd.DataFrame(list_of_ids, columns=['id', 'question', 'answer', 'choices', 'model_output', 'model'])

    # Convert choices list to string for grouping
    store_choices = df['choices'].tolist()
    df['choices'] = df['choices'].apply(lambda x: '; '.join(x) if isinstance(x, list) else str(x))

    # Group and aggregate outputs
    df_grouped = df.groupby(['id', 'question', 'answer', 'choices']).apply(
        lambda x: dict(zip(x['model'], x['model_output']))
    ).reset_index(name='model_outputs')

    # Use the store choices for the grouped dataframe
    #df_grouped['choices'] = store_choices

    # Filter for questions answered by all models
    counts = df.groupby('id')['model'].nunique()
    ids_all_models = counts[counts == len(models_list)].index.tolist()
    questions_answered_by_llm = df[df['id'].isin(ids_all_models)]
    questions_answered_by_llm.reset_index(drop=True, inplace=True)

    # Drop duplicates and unnecessary columns
    questions_answered_by_llm_unique = questions_answered_by_llm.drop(columns=['model_output', 'model']).drop_duplicates()

    # Merge with grouped outputs
    merged_df = questions_answered_by_llm_unique.merge(df_grouped, on=['id', 'question', 'answer', 'choices'])

    # Convert to dict for JSON output
    result = merged_df.to_dict(orient='records')

    # Save the filtered questions to a new JSON file
    with open(os.path.join(args.dest_folder, "questions_answered_correctly_by_all_llms_" + benchmark_name + '.json'), 'w') as output_file:
        json.dump(result, output_file, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get questions answered by LLMs.")
    parser.add_argument(
        '--outputs_path', type=str, required=True, help='Path to the JSON file containing the data.',
    )
    parser.add_argument(
        '--dest_folder', type=str, default='',
        help='Path to save the output JSON file with questions answered by LLM.'
    )
    args = parser.parse_args()

    main(args)
