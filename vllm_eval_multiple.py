import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse

# ——————————————
# Config (user-defined)
# ——————————————
# model_path = "Qwen/Qwen2.5-Math-1.5B"
# model_path = "./qwen3b-it-SFT-boxed/checkpoint-250"

# model_path = "./0422-qwen3b-question-only-no-format-weighted-sft-cp175/checkpoint-125"
model_path = "./vanilla_trainer/0425-base-self-distill/checkpoint-41718"

# csv_path = "0421-qwen3b-question-only-no-format-online-sft-cp50.csv"

# model_path ="Qwen/Qwen2.5-3B-instruct"
# model_path = "Qwen/Qwen2.5-3B"

# FOLLOWING THE SOBER PAPER
num_trials = 10              # Number of full runs over the dataset
temperature = 0.8
top_p = 0.9

# ——————————————
# Prompt Template
# ——————————————
SYSTEM_PROMPT_1="<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SYSTEM_PROMPT_2="{prompt}"
SYSTEM_PROMPT_3="A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {prompt}\nAssistant: <think>"
SYSTEM_PROMPT_4="\n<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n\n"

SYSTEM_PROMPT = SYSTEM_PROMPT_2

# ——————————————
# Reward functions
# ——————————————
def reward_with_format(s, gt):
    answer = last_boxed_only_string(s)
    if answer is None:
        return 0
    try:
        return int(verify(parse(answer), parse(gt)))
    except:
        return 0

def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

# ——————————————
# Load dataset
# ——————————————
test_ds = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]
base_prompts = [SYSTEM_PROMPT.format(prompt=ex["problem"]) for ex in test_ds]
print(base_prompts[0])
ground_truths = [last_boxed_only_string(ex["solution"]) for ex in test_ds]

# Duplicate for each trial
all_prompts = base_prompts * num_trials
all_ground_truths = ground_truths * num_trials  # Will align index-wise

# ——————————————
# Generate responses
# ——————————————
llm = LLM(model=model_path)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=3000,
    n=1
)

print(f"Generating {len(all_prompts)} completions ({num_trials} trials × {len(base_prompts)} prompts)...")
outputs = llm.generate(all_prompts, sampling_params)

# ——————————————
# Compute rewards and reshape
# ——————————————
rewards_with_format = []
rewards_without_format = []

for out, gt in zip(outputs, all_ground_truths):
    resp = out.outputs[0].text
    rewards_with_format.append(reward_with_format(resp, gt))
    rewards_without_format.append(reward_without_format(resp, gt))

# Reshape to [num_trials, num_questions]
rewards_with_format = np.array(rewards_with_format).reshape(num_trials, -1)
rewards_without_format = np.array(rewards_without_format).reshape(num_trials, -1)

# ——————————————
# Summary statistics
# ——————————————
trial_means_wf = rewards_with_format.mean(axis=1)
trial_means_wof = rewards_without_format.mean(axis=1)

print("\n========== FINAL SUMMARY ==========")
print(f"Meas (with format):                 {trial_means_wf}")
print(f"Mean of means (with format):        {np.mean(trial_means_wf):.3f}")
print(f"Standard deviation of means (with): {np.std(trial_means_wf):.6f}\n")

print(f"Means (without format):                {trial_means_wof}")
print(f"Mean of means (without format):        {np.mean(trial_means_wof):.3f}")
print(f"Standard deviation of means (without): {np.std(trial_means_wof):.6f}")

# ——————————————
# Save responses and lengths
# ——————————————
responses = [out.outputs[0].text for out in outputs]
response_lengths = [len(r) for r in responses]

df = pd.DataFrame({
    "prompt": all_prompts,
    "ground_truth": all_ground_truths,
    "response": responses,
    "response_length": response_lengths
})
# df.to_csv(csv_path, index=False)

# ——————————————
# Response length statistics
# ——————————————
response_lengths_array = np.array(response_lengths).reshape(num_trials, -1)
trial_mean_lengths = response_lengths_array.mean(axis=1)

print("\n========== RESPONSE LENGTH SUMMARY ==========")
print(f"Mean response lengths per trial:       {trial_mean_lengths}")
print(f"Mean of means (response length):       {np.mean(trial_mean_lengths):.3f}")
print(f"Standard deviation of mean lengths:    {np.std(trial_mean_lengths):.6f}")
