export VLLM_USE_V1=1

model_name="Qwen/Qwen2.5-3B"

vllm serve Qwen/Qwen2.5-3B \
	--max-model-len 12000 \
	--seed 42 \
	--gpu-memory-utilization 0.9 \
	--max-seq-len-to-capture 12000 \
	--override-generation-config '{"temperature": 0}' \
	--enable-reasoning \
	--reasoning-parser 'deepseek_r1'
