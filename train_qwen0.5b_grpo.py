# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="test")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen/Qwen2-0.5B", logging_steps=10, use_vllm=True, vllm_gpu_memory_utilization=0.6, bf16=True, bf16_full_eval=True)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()