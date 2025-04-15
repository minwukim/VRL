
export GPUS=$(($(nvidia-smi --list-gpus | wc -l) - 1))
export WANDB_PROJECT=sft_kkpro-math
nohup accelerate launch --config_file zero3.yaml --num_processes=$GPUS trainer_qwen.py --config grpoconfig.yaml  > training_log.out 2>&1 &

