
CUDA_VISIBLE_DEVICES=0
nohup trl vllm-serve --model "Qwen/Qwen2.5-3B" --gpu_memory_utilization 0.5 > vllm_log.out 2>&1 &

#CUDA_VISIBLE_DEVICES=1,2
#nohup accelerate launch --config_file zero3.yaml --num_processes=$GPUS trainer_qwen.py --config grpoconfig.yaml  > training_log.out 2>&1 &

