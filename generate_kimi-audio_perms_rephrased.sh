#!/bin/bash
#$ -o ./data/logs/generate_kimi-audio_perm-$TASK_ID.o
#$ -e ./data/logs/generate_kimi-audio_perm-$TASK_ID.e

export CUDA_HOME=/usr/local/share/cuda-12.1/
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) ### BUT cluster, here put your free GPU

export PYTHONPATH="./src/Kimi-Audio/:$PYTHONPATH"

# Set maximum execution time to 48 hours
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

# Permuation index
PERM_IDX=0

# Other options
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


#################################################
# Paths and filenames
#################################################

DATASETS=("MMAU" "MMAR" "MMSU")
PERMUTATION_TYPE="perm-mix" # Possibilities: question_rp, answer_rp, distractors_rp

if [ "$PERMUTATION_TYPE" = "question_rp" ]; then
    REPHRASE_TYPES=("rephrased-q-with-da_gemini-2.5-flash-1" "rephrased-q-with-da_gemini-2.5-flash-2" "rephrased-q-with-da_gemini-2.5-flash-3" "rephrased-q_gemma-3-12b-it" "rephrased-q-with-a_gemma-3-12b-it" "rephrased-q-with-d_gemma-3-12b-it")
elif [ "$PERMUTATION_TYPE" = "answer_rp" ]; then
    REPHRASE_TYPES=("rephrased-a-with-qd_gemini-2.5-flash-1" "rephrased-a-with-qd_gemini-2.5-flash-2" "rephrased-a-with-qd_gemini-2.5-flash-3" "rephrased-a-with-qd_gemma-3-12b-it" "rephrased-a-with-d_gemma-3-12b-it" "rephrased-a-with-q_gemma-3-12b-it")
elif [ "$PERMUTATION_TYPE" = "distractors_rp" ]; then
    REPHRASE_TYPES=("rephrased-d-with-qa_gemini-2.5-flash-1" "rephrased-d-with-qa_gemini-2.5-flash-2" "rephrased-d-with-qa_gemini-2.5-flash-3" "rephrased-d-with-a_gemma-3-12b-it" "rephrased-d-with-q_gemma-3-12b-it" "rephrased-d-with-qa_gemma-3-12b-it")
elif [ "$PERMUTATION_TYPE" = "perm-mix" ]; then
    REPHRASE_TYPES=("perm-mix-s-0" "perm-mix-s-2025" "perm-mix-s-7764" "perm-mix-s-111111" "perm-mix-s-441919" "perm-mix-s-943597")
else
    echo "Unknown permutation type: $PERMUTATION_TYPE"
    exit 1
fi


for DATASET in "${DATASETS[@]}"; do
    for REPHRASE_TYPE in "${REPHRASE_TYPES[@]}"; do
        echo "**************************************************"
        echo "Starting processing for dataset: $DATASET with rephrase type: $REPHRASE_TYPE"
        
        # If dataset is MMAU, use the following line
        if [ "$DATASET" = "MMAU" ]; then
            echo "Using MMAU dataset"
            RESULTS_PATH=$BASE_PATH"/results/mmau-v05.15.25/outputs/${PERMUTATION_TYPE}_${MODEL_SHORT_NAME}/"
            DATASET_AUDIO="./data/MMAU-v05.15.25/test-mini-audios"
            BASE_NAME="mmau-test-mini"
            DATASET_FILENAME="./data/MMAU-v05.15.25/${REPHRASE_TYPE}_${BASE_NAME}.json"
        elif [ "$DATASET" = "MMAR" ]; then
            echo "Using MMAR dataset"
            RESULTS_PATH=$BASE_PATH"/results/mmar/outputs/${PERMUTATION_TYPE}_${MODEL_SHORT_NAME}/"
            DATASET_AUDIO="./data/MMAR/audio"
            BASE_NAME="MMAR-meta"
            DATASET_FILENAME="./data/MMAR/${REPHRASE_TYPE}_${BASE_NAME}.json"
        elif [ "$DATASET" = "MMSU" ]; then
            echo "Using MMSU dataset"
            RESULTS_PATH=$BASE_PATH"/results/mmsu/outputs/${PERMUTATION_TYPE}_${MODEL_SHORT_NAME}/"
            DATASET_AUDIO="./data/MMSU/audio"
            BASE_NAME="mmsu"
            DATASET_FILENAME="./data/MMSU/question/${REPHRASE_TYPE}_${BASE_NAME}.json"
        else
            echo "Unknown dataset: $DATASET"
            exit 1
        fi

        # Create results path if it does not exist
        mkdir -p $RESULTS_PATH

        #################################################

        # Generate a log file with the output
        LOG_FILE="${BASE_PATH}/logs/generate_outputs_${MODEL_SHORT_NAME}_${DATASET}_${REPHRASE_TYPE}.log"
        OUTPUT_FILENAME="${DATASET}_${MODEL_SHORT_NAME}_${REPHRASE_TYPE}.json"

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

        echo "***************************************************"

    done
done
