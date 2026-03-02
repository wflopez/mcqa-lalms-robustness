#!/bin/bash
#$ -o ./data/logs/generate_outputs_af2_nb_perm-$TASK_ID.o
#$ -e ./data/logs/generate_outputs_af2_nb_perm-$TASK_ID.e

export PYTHONPATH="./src/audio_flamingo_3/:$PYTHONPATH"

# CUDA: check in the machine
#export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

# Set maximum execution time to 5 hours
ulimit -t 10800

# Removes the maximum file size that can be created
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

# Limit the number of processes
ulimit -u 4096

# Not use remote data
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Working directory
BASE_PATH=.

# Move to the working directory
cd $BASE_PATH

# Activate the virtual environment
source src/audio_flamingo_3/.venv/bin/activate


################################################
# Running configs
################################################

# Seed for reproducibility
SEED=2025 # Is used to truncate choices when they are more than four

# Permuation index: from argument
PERM_IDX=$1

# Dataset to use, can be "MMAU", "MMAR" or "MMSU"
DATASET="MMSU"
#DATASET="MMAU" 
#DATASET="MMAU" # Change this to "MMAR" or "MMSU" as needed

echo "Running with seed:" $SEED
echo "Permutation index:" $PERM_IDX
echo "Using dataset:" $DATASET

#################################################
# Paths and filenames
#################################################

# If dataset is MMAU, use the following line
if [ "$DATASET" = "MMAU" ]; then
    echo "Using MMAU dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmau-v05.15.25/outputs/perms_af3/"
    DATASET_FILENAME="./data/MMAU-v05.15.25/mmau-test-mini.json"
    DATASET_AUDIO="./data/MMAU-v05.15.25/test-mini-audios"
elif [ "$DATASET" = "MMAR" ]; then
    echo "Using MMAR dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmar/outputs/perms_af3/"
    DATASET_FILENAME="./data/MMAR/MMAR-meta.json"
    DATASET_AUDIO="./data/MMAR/audio"
elif [ "$DATASET" = "MMSU" ]; then
    echo "Using MMSU dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmsu/outputs/perms_af3/"
    DATASET_FILENAME="./data/MMSU/question/mmsu.json"
    DATASET_AUDIO="./data/MMSU/audio"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Create results path if it does not exist
mkdir -p $RESULTS_PATH

#################################################
# AF2 config file: info about the model
#################################################
CONFIG_FILE=$BASE_PATH"/src/audio_flamingo_3/config/inference.yaml"

#################################################
# Do not change anything below this line
#################################################

LOG_FILE="${BASE_PATH}/logs/generate_outputs_af3_${DATASET}_perm-${PERM_IDX}_s${SEED}.log"
OUTPUT_FILENAME="${DATASET}_af3_perm-${PERM_IDX}.json"

{
    echo "Running on machine: $(hostname)"
    echo "Started on: $(date)"
    echo "Output filename: $OUTPUT_FILENAME"
} > "$LOG_FILE"


python -u src/generate_outputs_af3_perm.py \
    --input $DATASET_FILENAME \
    --audio_path $DATASET_AUDIO \
    --output_folder $RESULTS_PATH \
    --output_filename $OUTPUT_FILENAME \
    --seed $SEED \
    --config $CONFIG_FILE \
    --perm_idx $PERM_IDX >> "$LOG_FILE" 2>&1

echo "Finished on: $(date)" >> "$LOG_FILE"