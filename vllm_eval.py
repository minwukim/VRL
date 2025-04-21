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
csv_path   = "qwen_3b_base_eval.csv"

SYSTEM_PROMPT = """
<|im_start|>system
Please reason step by step, and put your final answer within \\boxed{{}}.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
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
# Load data
# ——————————————
test_ds = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]

# ——————————————
# Prepare model + sampling
# ——————————————
llm = LLM(model=model_path)
sampling_params = SamplingParams(temperature=0.0, max_tokens=2560, top_p=1.0)

# ——————————————
# Inference + scoring loop
# ——————————————
records = []
prompts = []

for ex in test_ds:
    prompt = SYSTEM_PROMPT.format(prompt=ex["problem"])
    gt = last_boxed_only_string(ex["solution"])
    records.append({"problem": ex["problem"], "ground_truth": gt})
    prompts.append(prompt)

outs = llm.generate(prompts, sampling_params)

results = []

for rec, out in zip(records, outs):
    a1 = out.outputs[0].text
    rec["response"] = a1

    # Compute rewards
    gt = rec["ground_truth"]
    rec["reward_with_format"] = reward_with_format(a1, gt)
    rec["reward_without_format"] = reward_without_format(a1, gt)

    results.append(rec)

# ——————————————
# Save results to CSV
# ——————————————
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)

# ——————————————
# Print summary stats
# ——————————————
mean_with_format = df["reward_with_format"].mean()
mean_without_format = df["reward_without_format"].mean()

count_with_format = df["reward_with_format"].value_counts().sort_index()
count_without_format = df["reward_without_format"].value_counts().sort_index()

print(f"Mean Reward (with format): {mean_with_format:.3f}")
print(f"Mean Reward (without format): {mean_without_format:.3f}\n")

print("Reward With Format - Distribution:")
print(count_with_format.to_string())

print("\nReward Without Format - Distribution:")
print(count_without_format.to_string())
