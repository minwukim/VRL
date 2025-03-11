import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from datasets import load_dataset
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""


def reward_func(completions, answer, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"<answer>([\s\S]*)</answer>", completion) for completion in completions] 
    completions = [match.group(1) if match else "" for match in matches]
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if verify(parse(c), parse(gt))  else 0.0 for c, gt in zip(contents, answer)]

def extract_boxed_answer(solution):
    return last_boxed_only_string(solution)

def get_dataset():
    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": extract_boxed_answer(x["solution"]),
        "level": x["level"]
        })

    
    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["answer"],
        "level": x["level"]
        })
    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test

train, test = get_dataset()

model_name = "Qwen/Qwen2.5-3B"
output_dir = "outputs/qwen2.5-3b-grpo"
run_name = "qwen3b"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_steps = 25,
    lr_scheduler_type='constant_with_warmup',
    logging_steps=1,
    bf16=True,
    bf16_full_eval=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=4096,
    num_train_epochs=1,
    save_steps=200,
    max_grad_norm=0.1,
    report_to="wandb",
    use_vllm=True,
    vllm_max_model_len=5000,
    max_steps=100,
    log_completions=True
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=train,
    eval_dataset=test
)
trainer.train()
