import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse

# ——————————————
# Config
# ——————————————
model_path = "Qwen/Qwen2.5-3B"
csv_path   = "qwen_3b_dual_prompt_eval.csv"

SYSTEM_PROMPT_1 = """
<|im_start|>system
Please reason step by step, and put your final answer within \\boxed{{}}.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

SYSTEM_PROMPT_2 = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

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
# Load data
# ——————————————
test_ds = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]

# ——————————————
# Prepare model + sampling
# ——————————————
llm = LLM(model=model_path)
sampling_params = SamplingParams(temperature=0.0, max_tokens=2560, top_p=1.0)

# ——————————————
# Generate prompts
# ——————————————
records = []
prompts_1, prompts_2 = [], []

for ex in test_ds:
    problem = ex["problem"]
    gt = last_boxed_only_string(ex["solution"])
    p1 = SYSTEM_PROMPT_1.format(prompt=problem)
    p2 = SYSTEM_PROMPT_2.format(prompt=problem)
    records.append({
        "problem": problem,
        "ground_truth": gt,
        "prompt_1": p1,
        "prompt_2": p2
    })
    prompts_1.append(p1)
    prompts_2.append(p2)

# ——————————————
# Generate outputs for both prompts
# ——————————————
outs_1 = llm.generate(prompts_1, sampling_params)
outs_2 = llm.generate(prompts_2, sampling_params)

# ——————————————
# Score and collect results
# ——————————————
for rec, o1, o2 in zip(records, outs_1, outs_2):
    r1 = o1.outputs[0].text
    r2 = o2.outputs[0].text
    gt = rec["ground_truth"]

    rec["response_1"] = r1
    rec["response_2"] = r2

    rec["reward1_format"] = reward_with_format(r1, gt)
    rec["reward1_noformat"] = reward_without_format(r1, gt)

    rec["reward2_format"] = reward_with_format(r2, gt)
    rec["reward2_noformat"] = reward_without_format(r2, gt)

# ——————————————
# Export to CSV
# ——————————————
df = pd.DataFrame(records)
df.to_csv(csv_path, index=False)

# ——————————————
# Print summary stats
# ——————————————
print("========== REWARD DISTRIBUTIONS ==========\n")

def print_reward_counts(col_name):
    counts = df[col_name].value_counts().sort_index()
    print(f"{col_name} distribution:\n{counts.to_string()}\n")

print_reward_counts("reward1_format")
print_reward_counts("reward2_format")
print_reward_counts("reward1_noformat")
print_reward_counts("reward2_noformat")

# ——————————————
# Print mismatched rewards
# ——————————————
print("========== MISMATCHED REWARDS ==========\n")

# 1. Format-sensitive mismatch
mismatch_format = df[df["reward1_format"] != df["reward2_format"]]
print(f"Mismatched reward_with_format: {len(mismatch_format)} examples")
print(mismatch_format[["problem", "reward1_format", "reward2_format"]].head(5).to_string(index=False), "\n")

# 2. Format-agnostic mismatch
mismatch_noformat = df[df["reward1_noformat"] != df["reward2_noformat"]]
print(f"Mismatched reward_without_format: {len(mismatch_noformat)} examples")
print(mismatch_noformat[["problem", "reward1_noformat", "reward2_noformat"]].head(5).to_string(index=False))
