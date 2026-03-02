#!/bin/bash

export GEMINI_API_KEY="YOUR_API_KEY_HERE"


##################################################
# Running configs
##################################################

MODEL_NAME="gemini-2.5-flash"

# Dataset to use, can be "MMAU", "MMAR" or "MMSU"

#DATASET="MMAR"
#DATASET="MMAU"
#DATASET="MMSU"
#DATASET="MMAU-Pro"

DATASETS=("MMAR" "MMSU")

# Running vars
BATCH_SIZE=20

# Whether to include the answer for rephrasing
INCLUDE_ANSWER=false
INCLUDE_DISTRACTORS=true

args=""

if [ "$INCLUDE_ANSWER" = true ] ; then
    args+=" --include_gt"
fi

if [ "$INCLUDE_DISTRACTORS" = true ] ; then
    args+=" --include_distractors"
fi

##################################################
# Paths and filenames
##################################################

MODEL_SHORT_NAME=${MODEL_NAME##*/}


for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

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

    echo "Dataset path: $DATASET_PATH"
    echo "Dataset filename: $DATASET_FILENAME"
    echo "Model name: $MODEL_NAME"
    echo "Batch size: $BATCH_SIZE"
    echo "Include answer: $INCLUDE_ANSWER"
    echo "Include distractors: $INCLUDE_DISTRACTORS"
    echo "Args: $args"
    echo "Running rephrasing script..."

    # Run the rephrasing script with the specified parameters
    # Use -u to run the script in unbuffered mode for real-time output
    python -u src/scripts/rephrase_questions_gemini.py \
        --dataset_folder $DATASET_PATH \
        --dataset_filename $DATASET_FILENAME \
        --model_name $MODEL_NAME \
        $args
done

