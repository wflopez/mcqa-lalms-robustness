#!/bin/bash

# CUDA: check in the machine
#export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

# Set maximum execution time to 24 hours
ulimit -t 86400

# Removes the maximum file size that can be created
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

# Not use remote data
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false


# Working directory
BASE_PATH=.

# Move to the working directory
cd $BASE_PATH

# Activate the virtual environment
source .venv/bin/activate

##################################################
# Running configs
##################################################

# Models path 
HF_CACHE_DIR=$BASE_PATH"models/"

MODEL_NAME="google/gemma-3-12b-it"

# Dataset to use, can be "MMAU", "MMAR" or "MMSU"

DATASET="MMAR"
#DATASET="MMAU"
#DATASET="MMSU"
#DATASET="MMAU-Pro"

# Running vars
BATCH_SIZE=8
NUM_WORKERS=4
SEED=2025  # Seed for reproducibility

# Whether to include the answer for rephrasing
INCLUDE_ANSWER=false
# Whether to include distractors for rephrasing
INCLUDE_DISTRACTORS=true


args=""

if [ "$INCLUDE_ANSWER" = false ] ; then
    args+=" --not_include_gt"
fi

if [ "$INCLUDE_DISTRACTORS" = false ] ; then
    args+=" --not_include_distractors"
fi

##################################################
# Paths and filenames
##################################################

MODEL_SHORT_NAME=${MODEL_NAME##*/}

if [ "$DATASET" = "MMAU" ]; then
    echo "Using MMAU dataset"
    DATASET_PATH="./data/MMAU-v05.15.25/"
    DATASET_FILENAME="mmau-test-mini.json"
elif [ "$DATASET" = "MMAU-Pro" ]; then
    echo "Using MMAU-Pro dataset"
    DATASET_PATH="./"
    DATASET_FILENAME="mmau_combined.json"
elif [ "$DATASET" = "MMAR" ]; then
    echo "Using MMAR dataset"
    DATASET_PATH="./data/MMAR/"
    DATASET_FILENAME="MMAR-meta.json"
elif [ "$DATASET" = "MMSU" ]; then
    echo "Using MMSU dataset"
    DATASET_PATH="./data/MMSU/question/"
    DATASET_FILENAME="mmsu.json"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Generate a log file with the output
if [ "$INCLUDE_DISTRACTORS" = true ] ; then
    LOG_FILE="${BASE_PATH}/logs/rephrase-q-with-d_${DATASET}_${MODEL_SHORT_NAME}.log"
elif [ "$INCLUDE_ANSWER" = true ] ; then
    LOG_FILE="${BASE_PATH}/logs/rephrase-q-with-a_${DATASET}_${MODEL_SHORT_NAME}.log"
else
    LOG_FILE="${BASE_PATH}/logs/rephrase-q_${DATASET}_${MODEL_SHORT_NAME}.log"
fi 

{
    echo "Running on machine: $(hostname)"
    echo "Started on: $(date)"
} > "$LOG_FILE"


python -u src/scripts/rephrase_questions.py \
    --model_name $MODEL_NAME \
    --dataset_folder $DATASET_PATH \
    --dataset_filename $DATASET_FILENAME \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_tokens 512 \
    --seed $SEED \
    $args >> "$LOG_FILE" 2>&1

echo "Finished on: $(date)" >> "$LOG_FILE"
