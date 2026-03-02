#!/bin/bash

# CUDA: check in the machine
#export CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

# Set maximum execution time to 10 hours
ulimit -t 172800

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

# Model
#MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME="google/gemma-3-4b-it"
#MODEL_NAME="google/gemma-3-12b-it"
MODEL_NAME="google/gemma-3-27b-it"
#MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"


# Dataset to use, can be "MMAU", "MMAR" or "MMSU"

#DATASET="MMAR"
#DATASET="MMAU"
DATASET="MMSU"
#DATASET="MMAU-Pro"

# Running vars
BATCH_SIZE=2
NUM_WORKERS=4
SEED=2025  # Seed for reproducibility


##################################################
# Paths and filenames
##################################################

MODEL_SHORT_NAME=${MODEL_NAME##*/}

if [ "$DATASET" = "MMAU" ]; then
    echo "Using MMAU dataset"
    DATASET_PATH="./data/MMAU-v05.15.25/"
    DATASET_FILENAME="mmau-test-mini.json"
    OUTPUT_FOLDER=$BASE_PATH"/results/mmau-v05.15.25/outputs/${MODEL_SHORT_NAME}/"
elif [ "$DATASET" = "MMAU-Pro" ]; then
    echo "Using MMAU-Pro dataset"
    #DATASET_PATH="./data/mmau_pro/"
    #DATASET_PATH="./data/MMAU-Pro/music_2/"
    DATASET_PATH="./data/data/MMAU-PRO-SPEECH/latest/"
    #DATASET_FILENAME="concatenated.json"
    DATASET_FILENAME="mmau_pro.json"
    OUTPUT_FOLDER=$BASE_PATH"/results/mmau-pro/outputs/${MODEL_SHORT_NAME}/"
elif [ "$DATASET" = "MMAR" ]; then
    echo "Using MMAR dataset"
    DATASET_PATH="./data/MMAR/"
    DATASET_FILENAME="MMAR-meta.json"
    OUTPUT_FOLDER=$BASE_PATH"/results/mmar/outputs/${MODEL_SHORT_NAME}/"
elif [ "$DATASET" = "MMSU" ]; then
    echo "Using MMSU dataset"
    DATASET_PATH="./data/MMSU/question/"
    DATASET_FILENAME="mmsu.json"
    OUTPUT_FOLDER=$BASE_PATH"/results/mmsu/outputs/${MODEL_SHORT_NAME}/"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Create output folder if it does not exist
mkdir -p $OUTPUT_FOLDER

# Output path
OUTPUT_FILENAME=$DATASET"_"$MODEL_SHORT_NAME".json"

# Generate a log file with the output
LOG_FILE="${BASE_PATH}/logs/${DATASET}_${MODEL_SHORT_NAME}.log"

{
    echo "Running on machine: $(hostname)"
    echo "Started on: $(date)"
    echo "Output filename: $OUTPUT_FILENAME"
} > "$LOG_FILE"


python -u src/scripts/generate_outputs_llm.py \
    --model_name $MODEL_NAME \
    --dataset_folder $DATASET_PATH \
    --dataset_filename $DATASET_FILENAME \
    --output_folder $OUTPUT_FOLDER \
    --output_filename $OUTPUT_FILENAME \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_tokens 512 \
    --seed $SEED >> "$LOG_FILE" 2>&1

echo "Finished on: $(date)" >> "$LOG_FILE"
