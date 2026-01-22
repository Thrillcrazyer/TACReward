# List of datasets to evaluate
TEST_NAMES=(
    "zwhe99/MATH"
    "zwhe99/aime90"
    "math-ai/aime25"
    "zwhe99/simplerl-minerva-math"
    "zwhe99/simplerl-OlympiadBench"
    "Quadyun/Korean_SAT_MATH"
    "opencompass/LiveMathBench"
)


# Subset configuration per dataset
declare -A SUBSETS
SUBSETS=(
    ["Quadyun/Korean_SAT_MATH"]="2025_calculus"
    ["opencompass/LiveMathBench"]="v202505_all_en"
)

# Model list and corresponding paths
declare -A MODELS
MODELS=(
    ["TACReward"]="./Result/Qwen-1.5B_THIP_1125/checkpoint-1000"
    ["DeepSeek-R1-Distill-Qwen-1.5B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# Common settings
COMMON_ARGS="
    --chat_template_name r1-distill-qwen
    --system_prompt_name custom
    --bf16 True
    --tensor_parallel_size 1
    --max_model_len 16384
    --temperature 0.6
    --top_p 0.95
    --n 1
"

# Iterate and evaluate
for TEST_NAME in "${TEST_NAMES[@]}"; do
    echo "🔹 Evaluating dataset: $TEST_NAME"
    
    # Set subset argument
    SUBSET_ARG=""
    if [[ -n "${SUBSETS[$TEST_NAME]}" ]]; then
        SUBSET_ARG="--subset ${SUBSETS[$TEST_NAME]}"
    fi
    
    for MODEL_KEY in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$MODEL_KEY]}
        OUTPUT_DIR="./eval_results/${MODEL_KEY}/$(basename $TEST_NAME)"
        LOG_FILE="log_${MODEL_KEY}_$(basename $TEST_NAME).txt"

        echo "🚀 Running ${MODEL_KEY} on ${TEST_NAME}"
        mkdir -p "$OUTPUT_DIR"

        # Execute with environment variables
        CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python evaluate.py \
            --base_model "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --data_id "$TEST_NAME" \
            $SUBSET_ARG \
            $COMMON_ARGS \

        echo "✅ Finished ${MODEL_KEY} on ${TEST_NAME}, log saved to $LOG_FILE"
        echo "-----------------------------------------"
    done
done

echo "evaluations completed"