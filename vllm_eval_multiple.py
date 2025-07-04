import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
# from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse

# ——————————————
# Config (user-defined)
# ——————————————
# model_path = "Qwen/Qwen2.5-Math-1.5B"

 
# model_path ="Qwen/Qwen2.5-3B-instruct"
# model_path = "Qwen/Qwen2.5-3B"
model_path = "Qwen/Qwen2.5-Math-1.5B"


model_path = "./0702-1.5B-1to64/checkpoint-325"
# model_path = "Qwen/Qwen2.5-7B"
# model_path = "./0421-qwen3b-question-only-no-format/checkpoint-150"
# model_path = "./0627-3B-7500-24-gen/checkpoint-175"  # Path to the model



# FOLLOWING THE SOBER PAPERR
num_trials = 1              # Number of full runs over the dataset
# temperature = 0.9
# top_p = 0.85
temperature = 0.9
top_p = 1
top_k = 50


# ——————————————
# Prompt Template
# ——————————————
SYSTEM_PROMPT_1="<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
SYSTEM_PROMPT_2="{prompt}"
SYSTEM_PROMPT_3="A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {prompt}\nAssistant: <think>"
SYSTEM_PROMPT_4="<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n\n"
SYSTEM_PROMPT_5="{prompt} [SEP] "
SYSTEM_PROMPT_6 = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it."
    "The Assistant  first thinks about the reasoning process in the mind and then provides the User with the answer."
    "The reasoning process is enclosed within <think> </think> and answer is enclosed with in <answer> </answer> tages, respectively,"
    " i.e., <think> reasoning process here </think> <answer> answer here </answer>./n"
    "User: {prompt}/nAssitant: <think>"
)

SYSTEM_PROMPT = SYSTEM_PROMPT_2

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

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
# test_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")

base_prompts = [SYSTEM_PROMPT.format(prompt=ex["problem"]) for ex in test_ds]
print(base_prompts[0])
ground_truths = [last_boxed_only_string(ex["solution"]) for ex in test_ds]

# Duplicate for each trial
all_prompts = base_prompts * num_trials
all_ground_truths = ground_truths * num_trials  # Will align index-wise

# ——————————————
# Generate responses
# ——————————————
llm = LLM(model=model_path, max_model_len=4000, tensor_parallel_size=1)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_tokens=4000,
    n=1,
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
