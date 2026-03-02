#! /bin/bash

# First, calculate performance for each permutation, dataset, and model
./src/other_scripts/calculate_performance_perm.sh


# Then, calculate robustness results for each permutation, dataset, and model
./src/other_scripts/calculate_robustness_results.sh

# Finally, generate overall results for each permutation type
python src/scripts/generate_overall_results.py