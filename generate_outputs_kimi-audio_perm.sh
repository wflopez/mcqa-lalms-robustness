#!/bin/bash
#$ -o ./data/logs/generate_outputs_af2_nb_perm-$TASK_ID.o
#$ -e ./data/logs/generate_outputs_af2_nb_perm-$TASK_ID.e

export CUDA_HOME=/usr/local/share/cuda-12.1/
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) ### BUT cluster, here put your free GPU

export PYTHONPATH="./src/Kimi-Audio/:$PYTHONPATH"

# Set maximum execution time to 4 hours
ulimit -t 18000

# Removes the maximum file size that can be created
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

# Not use remote data
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false


# Move to the working directory
BASE_PATH=.

# Move to the working directory
cd $BASE_PATH

# Activate the virtual environment
source src/Kimi-Audio/.venv/bin/activate


################################################
# Running configs
################################################

# Seed for reproducibility
SEED=2025 # Is used to truncate choices when they are more than four

# Model name and short name
MODEL_NAME="kimi-audio" # Change this to the model you want to use
MODEL_SHORT_NAME=${MODEL_NAME##*/}

#DATASET="MMAR" # Dataset to use, can be "MMAU", "MMAR" or "MMSU"
#DATASET="MMAR" 
DATASET="MMSU"
#DATASET="MMAU"

# If dataset is MMAU, use the following line
if [ "$DATASET" = "MMAU" ]; then
    echo "Using MMAU dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmau-v05.15.25/outputs/perms_${MODEL_SHORT_NAME}/"
    DATASET_FILENAME="./data/MMAU-v05.15.25/mmau-test-mini.json"
    DATASET_AUDIO="./data/MMAU-v05.15.25/test-mini-audios"
elif [ "$DATASET" = "MMAU-Pro" ]; then
    echo "Using MMAU-Pro dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmau-pro/outputs/perms_${MODEL_SHORT_NAME}/"
    DATASET_FILENAME="./mmau_combined.json"
    DATASET_AUDIO="./data/data/MMAU-PRO-SPEECH/latest/data"
elif [ "$DATASET" = "MMAR" ]; then
    echo "Using MMAR dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmar/outputs/perms_${MODEL_SHORT_NAME}/"
    DATASET_FILENAME="./data/MMAR/MMAR-meta.json"
    DATASET_AUDIO="./data/MMAR/audio"
elif [ "$DATASET" = "MMSU" ]; then
    echo "Using MMSU dataset"
    RESULTS_PATH=$BASE_PATH"/results/mmsu/outputs/perms_${MODEL_SHORT_NAME}/"
    DATASET_FILENAME="./data/MMSU/question/mmsu.json"
    DATASET_AUDIO="./data/MMSU/audio"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Create results path if it does not exist
mkdir -p $RESULTS_PATH

# Permuation index
PERM_IDX=$1
#PERM_IDX=0

if [ -z "$PERM_IDX" ]; then
    echo "Permutation index not provided. Usage: $0 <perm_idx>"
    exit 1
fi

zero_audio=false
reverse_options=false
extra_option=false

args=""

if [ "$reverse_options" = true ] ; then
    args+=" --reverse_options"
fi 
if [ "$zero_audio" = true ] ; then
    args+=" --zero_audio"
fi
if [ "$extra_option" = true ] ; then
    args+=" --extra_option"
fi

echo "Running with seed:" $SEED
echo "Permutation index:" $PERM_IDX

# Generate a log file with the output
LOG_FILE="${BASE_PATH}/logs/generate_outputs_${MODEL_SHORT_NAME}-${DATASET}-perm-${PERM_IDX}.log"
OUTPUT_FILENAME="${DATASET}_${MODEL_SHORT_NAME}_perm-${PERM_IDX}.json"

{
    echo "Running on machine: $(hostname)"
    echo "Started on: $(date)"
    echo "Output filename: $OUTPUT_FILENAME"
    echo "Permutation index: $PERM_IDX"
    echo "Running with seed: $SEED"
    echo "Model name: $MODEL_NAME"
    echo "Base path: $BASE_PATH"
    echo "Results path: $RESULTS_PATH"
    echo "Dataset filename: $DATASET_FILENAME"
    echo "Dataset audio path: $DATASET_AUDIO"
    echo "Arguments: $args"
} > "$LOG_FILE"


python -u src/scripts/generate_outputs_kimi_audio_perm.py \
    --input $DATASET_FILENAME \
    --audio_path $DATASET_AUDIO \
    --output_folder $RESULTS_PATH \
    --output_filename $OUTPUT_FILENAME \
    --seed $SEED \
    --cache_dir $BASE_PATH/models/ \
    --model_name $MODEL_NAME \
    --perm_idx $PERM_IDX \
    $args >> "$LOG_FILE" 2>&1

echo "Finished on: $(date)" >> "$LOG_FILE"