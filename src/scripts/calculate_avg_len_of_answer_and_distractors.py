import json
import copy
import tqdm
import random
import itertools

import argparse
import numpy as np


if __name__ == "__main__":

    ################################################
    ############# Configuration ####################
    ################################################

    # For every benchmark, generate a json file with all possible permutations of the models
    benchmarks = ["MMAU", "MMAR", "MMSU"]

    # Isolated permutations to consider
    rephrased_questions = [
        "rephrased-q_gemma-3-12b-it",
        "rephrased-q-with-a_gemma-3-12b-it",
        "rephrased-q-with-da_gemini-2.5-flash-1",
        "rephrased-q-with-da_gemini-2.5-flash-2",
        "rephrased-q-with-da_gemini-2.5-flash-3",
        "rephrased-q-with-d_gemma-3-12b-it"
    ]

    rephrased_answers = [
        "rephrased-a-with-d_gemma-3-12b-it",
        "rephrased-a-with-qd_gemini-2.5-flash-1",
        "rephrased-a-with-qd_gemini-2.5-flash-2",
        "rephrased-a-with-qd_gemini-2.5-flash-3",
        "rephrased-a-with-qd_gemma-3-12b-it",
        "rephrased-a-with-q_gemma-3-12b-it"
    ]

    rephrased_distractors = [
        "rephrased-d-with-a_gemma-3-12b-it",
        "rephrased-d-with-qa_gemini-2.5-flash-1",
        "rephrased-d-with-qa_gemini-2.5-flash-2",
        "rephrased-d-with-qa_gemini-2.5-flash-3",
        "rephrased-d-with-qa_gemma-3-12b-it",
        "rephrased-d-with-q_gemma-3-12b-it"
    ]

    ################################################
    ############# Main script ######################
    ################################################

    all_question_lengths = []
    all_answer_lengths = []
    all_distractor_lengths = []

    all_paraphrased_question_lengths = []
    all_paraphrased_answer_lengths = []
    all_paraphrased_distractor_lengths = []


    for benchmark in benchmarks:

        if benchmark == "MMAU":
            dataset_path = "/mnt/matylda4/xlopezw00/MMAU-v05.15.25/"
            original_file = "mmau-test-mini.json"
        elif benchmark == "MMAR":
            dataset_path = "/mnt/matylda4/xlopezw00/MMAR/"
            original_file = "MMAR-meta.json"
        elif benchmark == "MMSU":
            dataset_path = "/mnt/matylda4/xlopezw00/MMSU/question/"
            original_file = "mmsu.json"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        with open(dataset_path + original_file, "r") as f:
            original_data = json.load(f)

        print("*"*50)
        
        # Get the length of the answers in the original data
        answer_lengths = [len(str(item["answer"]).split()) for item in original_data if "answer" in item]
        avg_answer_length = np.mean(answer_lengths)
        print(f"Benchmark: {benchmark}, Average length of answers: {avg_answer_length:.2f} words")
        all_answer_lengths.extend(answer_lengths)

        # Now do the same for the rephrased answers
        for rephrased_answer in rephrased_answers:
            ra_filename = f"{dataset_path}{rephrased_answer}_{original_file}"

            with open(ra_filename, "r") as f:
                ra_data = json.load(f)
            ra_answer_lengths = [len(str(item["answer"]).split()) for item in ra_data if "answer" in item]
            ra_avg_answer_length = np.mean(ra_answer_lengths)
            print(f"  Rephrased answer: {rephrased_answer}, Average length of answers: {ra_avg_answer_length:.2f} words")
            all_paraphrased_answer_lengths.extend(ra_answer_lengths)
        print("-"*50)

        # Get the length of the distractors in the original data
        # For that get choices and remove the answer from them
        distractor_lengths = []
        for item in original_data:
            if "choices" in item and "answer" in item:
                distractor = [choice for choice in item["choices"] if choice != item["answer"]]
                distractor_lengths.extend([len(str(d).split()) for d in distractor])
        distractor_avg_length = np.mean(distractor_lengths)
        print(f"Benchmark: {benchmark}, Average length of distractors: {distractor_avg_length:.2f} words")
        all_distractor_lengths.extend(distractor_lengths)

        
        # Now do the same for the rephrased distractors
        for rephrased_distractor in rephrased_distractors:
            rd_filename = f"{dataset_path}{rephrased_distractor}_{original_file}"

            with open(rd_filename, "r") as f:
                rd_data = json.load(f)
            rd_distractor_lengths = []
            for item in rd_data:
                if "choices" in item and "answer" in item:
                    distractor = [choice for choice in item["choices"] if choice != item["answer"]]
                    rd_distractor_lengths.extend([len(str(d).split()) for d in distractor])
            rd_avg_distractor_length = np.mean(rd_distractor_lengths)
            all_paraphrased_distractor_lengths.extend(rd_distractor_lengths)
            print(f"  Rephrased distractor: {rephrased_distractor}, Average length of distractors: {rd_avg_distractor_length:.2f} words")
        
        print("*"*50)

        # Get the length of the questions in the original data
        question_lengths = [len(str(item["question"]).split()) for item in original_data if "question" in item]
        avg_question_length = np.mean(question_lengths)
        print(f"Benchmark: {benchmark}, Average length of questions: {avg_question_length:.2f} words")
        all_question_lengths.extend(question_lengths)

        # Now do the same for the rephrased questions
        for rephrased_question in rephrased_questions:
            rq_filename = f"{dataset_path}{rephrased_question}_{original_file}"

            with open(rq_filename, "r") as f:
                rq_data = json.load(f)
            rq_question_lengths = [len(str(item["question"]).split()) for item in rq_data if "question" in item]
            rq_avg_question_length = np.mean(rq_question_lengths)
            all_paraphrased_question_lengths.extend(rq_question_lengths)
            print(f"  Rephrased question: {rephrased_question}, Average length of questions: {rq_avg_question_length:.2f} words")
        print("-"*50)

    # Agregate: general statistics
    # Overall original lengths
    overall_avg_answer_length = np.mean(answer_lengths)
    overall_avg_distractor_length = np.mean(distractor_lengths)
    overall_avg_question_length = np.mean(question_lengths)

    print("="*50)
    print(f"Overall average length of answers: {overall_avg_answer_length:.2f} words")
    print(f"Overall average length of distractors: {overall_avg_distractor_length:.2f} words")
    print(f"Overall average length of questions: {overall_avg_question_length:.2f} words")
    print(f"Overall average length of choices (including answer): {(np.mean(all_answer_lengths) + np.mean(all_distractor_lengths))/2:.2f} words")
    print("="*50)

    # Overall paraphrased lengths:
    overall_avg_paraphrased_answer_length = np.mean(all_paraphrased_answer_lengths)
    overall_avg_paraphrased_distractor_length = np.mean(all_paraphrased_distractor_lengths)
    overall_avg_paraphrased_question_length = np.mean(all_paraphrased_question_lengths)
    print(f"Overall average length of paraphrased answers: {overall_avg_paraphrased_answer_length:.2f} words")
    print(f"Overall average length of paraphrased distractors: {overall_avg_paraphrased_distractor_length:.2f} words")
    print(f"Overall average length of paraphrased questions: {overall_avg_paraphrased_question_length:.2f} words") 
    print(f"Overall average length of paraphrased choices (including answer): {(np.mean(all_paraphrased_answer_lengths) + np.mean(all_paraphrased_distractor_lengths))/2:.2f} words")

