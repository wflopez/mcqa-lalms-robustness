import pandas as pd


if __name__ == "__main__":

    # Permutations
    permutations = ["perms", "question_rp", "answer_rp", "distractors_rp", "perm-mix"]
    #permutations = ["all_perms"]

    #models = ["kimi-audio", "Qwen2.5-Omni-7B", "af3", "af2"]
    models = ["af2", "af3", "Qwen2.5-Omni-7B", "kimi-audio"]
    benchmarks = ["mmau-v05.15.25", "mmar", "mmsu"]

    # Load the results
    results_folder = "results"


    for permutation in permutations:
        results = []
        for model in models:
            for benchmark in benchmarks:
                # Load the results for each model and benchmark
                df = pd.read_csv(f"{results_folder}/{benchmark}/{permutation}_{model}/robustness.tsv", header=0, sep="\t")
                df["model"] = model
                df["benchmark"] = benchmark
                if permutation == "perms":
                    df["permutation"] = "choices_ordering"
                else:
                    df["permutation"] = permutation
                results.append(df)

        # Concatenate all results
        all_results = pd.concat(results, ignore_index=True)

        # Save to CSV
        all_results.to_csv(f"{results_folder}/results_overall_{permutation}.csv", index=False, sep="\t")


