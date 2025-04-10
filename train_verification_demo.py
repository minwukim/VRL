import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, TrlParser

from GRPO_customed.GRPO_custom import VerificationGRPOTrainer, SwitchingGRPOTrainer

from datasets import load_dataset
from math_verify import verify, parse
from obsolete.custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

from dataclasses import dataclass
from typing import Optional

max_seq_length = 500
max_prompt_length = 500 + 500

model_name = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""


def reward_correct_and_format(completions, answer, first_completions= None, **kwargs):
    # Regular expression to capture content inside \boxed{}
    print("reward-completions:", completions[0])
    matches = [re.search(r"</think>\n?<answer>([\s\S]*)</answer>", completion) for completion in completions] 
    completions = [match.group(1) if match else "" for match in matches]
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    correct_with_format = [1.0 if verify(parse(c), parse(gt))  else 0.0 for c, gt in zip(contents, answer)]

    print("======================================")
    print("reward-answer:", answer[0])
    if first_completions is not None:
        print("reward-first-turn-completion:", first_completions[0])
    print("======================================")

    return correct_with_format


# def reward_correct_and_format(completions, answer, **kwargs):
#     # Regular expression to capture content inside \boxed{}
#     matches = [re.search(r"</think>\n?<answer>([\s\S]*)</answer>", completion) for completion in completions] 
#     completions = [match.group(1) if match else "" for match in matches]
#     matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
#     contents = [match.group(1) if match else "" for match in matches]
#     # Reward 1 if the content is the same as the ground truth, 0 otherwise
#     correct_with_format = [1.0 if verify(parse(c), parse(gt))  else 0.0 for c, gt in zip(contents, answer)]

#     print("======================================")
#     print("reward-completions:", completions[0])
#     print("reward-answer:", answer[0])
#     print("======================================")

#     return correct_with_format
def reward_correct(completions, answer, **kwargs):
    correct = [1.0 if verify(parse(c), parse(gt))  else 0.0 for c, gt in zip(completions, answer)]
    return correct

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


training_args = GRPOConfig(
    use_vllm = True,
    output_dir = "0321-purerl-qwen0.5b",
    bf16 = True,
    bf16_full_eval=True,
    vllm_gpu_memory_utilization=0.9,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length,
    run_name = "0321-purerl-qwen0.5b",
    report_to = "wandb", 
    do_eval=True,
    per_device_train_batch_size=4,
    num_generations = 4,
    gradient_accumulation_steps = 1,
    num_train_epochs = 8,
    logging_steps=1,
    gradient_checkpointing=True,
    save_strategy = "steps",
    save_steps = 200,
    eval_strategy="steps",
    eval_steps = 200,
    log_completions = True
)


# trainer = VerificationGRPOTrainer(
#     model=model_name,
#     reward_funcs=[reward_correct, reward_correct_and_format],
#     args = training_args,
#     train_dataset=train,
#     eval_dataset=test,
# )

trainer = SwitchingGRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct_and_format],
    args = training_args,
    train_dataset=train,
    eval_dataset=test,
)

# trainer = ReplicatedGRPOTrainer(
#     model=model_name,
#     reward_funcs=[reward_correct, reward_correct_and_format],
#     args = training_args,
#     train_dataset=train,
#     eval_dataset=test,
# )
trainer.train()
