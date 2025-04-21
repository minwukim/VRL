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
model_path = "Qwen/Qwen2.5-3B"
num_trials = 5              # Number of full runs over the dataset
temperature = 0.7
top_p = 0.9

# ——————————————
# Prompt Template
# ——————————————
SYSTEM_PROMPT="""
<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
"""

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
    max_tokens=2560,
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
print(f"Meas (with format):                 {reward_with_format}")
print(f"Mean of means (with format):        {np.mean(trial_means_wf):.3f}")
print(f"Standard deviation of means (with): {np.std(trial_means_wf):.6f}\n")

print(f"Means (without format):                {reward_without_format}")
print(f"Mean of means (without format):        {np.mean(trial_means_wof):.3f}")
print(f"Standard deviation of means (without): {np.std(trial_means_wof):.6f}")
