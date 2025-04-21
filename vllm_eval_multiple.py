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
prompts = [SYSTEM_PROMPT.format(prompt=ex["problem"]) for ex in test_ds]
ground_truths = [last_boxed_only_string(ex["solution"]) for ex in test_ds]

# ——————————————
# Prepare model
# ——————————————
llm = LLM(model=model_path)

# ——————————————
# Run multiple trials
# ——————————————
trial_means_with_format = []
trial_means_without_format = []

for trial in range(num_trials):
    print(f"Running generation trial {trial + 1}/{num_trials}")
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=2560,
        n=1  # One response per prompt
    )

    outputs = llm.generate(prompts, sampling_params)

    rewards_with_format = []
    rewards_without_format = []

    for out, gt in zip(outputs, ground_truths):
        resp = out.outputs[0].text
        rewards_with_format.append(reward_with_format(resp, gt))
        rewards_without_format.append(reward_without_format(resp, gt))

    trial_mean_wf = np.mean(rewards_with_format)
    trial_mean_wof = np.mean(rewards_without_format)

    trial_means_with_format.append(trial_mean_wf)
    trial_means_without_format.append(trial_mean_wof)

# ——————————————
# Final summary
# ——————————————
print("\n========== FINAL SUMMARY ==========")
print(f"Mean of means (with format):     {np.mean(trial_means_with_format):.3f}")
print(f"Variance of means (with format): {np.var(trial_means_with_format):.6f}\n")

print(f"Mean of means (without format):     {np.mean(trial_means_without_format):.3f}")
print(f"Variance of means (without format): {np.var(trial_means_without_format):.6f}")
