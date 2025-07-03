#!/bin/bash

# Define model paths and their corresponding GPU assignments
declare -a models=(
    "Qwen/Qwen2.5-Math-1.5B"
    "./0702-1.5B-1to64/checkpoint-100"
    "./0702-1.5B-1to64/checkpoint-200"
    "./0702-1.5B-1to64/checkpoint-300"
    "./0702-1.5B-1to64/checkpoint-400"
)

for i in "${!models[@]}"; do
    gpu_id=$i
    model_path="${models[$i]}"
    save_name=$(basename "$model_path" | tr '/' '-' | tr '_' '-')
    log_file="logs/${save_name}.log"
    save_path="outputs/${save_name}.csv"

    mkdir -p logs outputs

    echo "Launching model $model_path on GPU $gpu_id..."
    
    nohup setsid bash -c "
        CUDA_VISIBLE_DEVICES=$gpu_id python run_math_eval.py \
        --model_path \"$model_path\" \
        --save_path \"$save_path\"
    " > "$log_file" 2>&1 &
done
