import re
import pandas as pd
import numpy as np
import argparse
from vllm import LLM, SamplingParams
from math_verify import verify, parse

# ——————————————
# Parse command-line arguments
# ——————————————
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--csv_path", type=str, default="math_base_model_test_question_solution_hit.csv")
args = parser.parse_args()

model_path = args.model_path
save_path = args.save_path
csv_path = args.csv_path

temperature = 0.9
top_p = 1
top_k = 50

# ——————————————
# Helper: last boxed string extractor
# ——————————————
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
    return string[idx:right_brace_idx + 1] if right_brace_idx else None

# ——————————————
# Reward function (no format)
# ——————————————
def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

# ——————————————
# Load CSV
# ——————————————
df = pd.read_csv(csv_path)
base_prompts = df["prompt"].tolist()
ground_truths = [last_boxed_only_string(gt) for gt in df["ground_truth"].tolist()]
question_indices = df["question_index"].tolist()
hits = df["hit"].tolist()

# ——————————————
# Generate responses
# ——————————————
print(f"[INFO] Starting generation using model: {model_path}")
llm = LLM(model=model_path, max_model_len=4000, tensor_parallel_size=1)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_tokens=4000,
    n=1,
)

print(f"[INFO] Generating {len(base_prompts)} completions...")
outputs = llm.generate(base_prompts, sampling_params)
responses = [out.outputs[0].text for out in outputs]

# ——————————————
# Evaluate rewards
# ——————————————
print("[INFO] Evaluating rewards...")
rewards = [reward_without_format(r, gt) for r, gt in zip(responses, ground_truths)]

# ——————————————
# Make output DataFrame
# ——————————————
out_df = pd.DataFrame({
    "question_index": question_indices,
    "prompt": base_prompts,
    "ground_truth": ground_truths,
    "response": responses,
    "reward": rewards,
    "hit": hits
})

# ——————————————
# Accuracy Statistics
# ——————————————
def accuracy_stats(df, min_hit=None, max_hit=None):
    subset = df
    if min_hit is not None:
        subset = subset[subset['hit'] > min_hit]
    if max_hit is not None:
        subset = subset[subset['hit'] <= max_hit]
    total = len(subset)
    correct = subset['reward'].sum()
    percentage = (correct / total * 100) if total > 0 else 0.0
    return correct, total, percentage

overall_correct, overall_total, overall_pct = accuracy_stats(out_df)
acc16_correct, acc16_total, acc16_pct = accuracy_stats(out_df, max_hit=16)
acc32_correct, acc32_total, acc32_pct = accuracy_stats(out_df, max_hit=32)
acc64_correct, acc64_total, acc64_pct = accuracy_stats(out_df, max_hit=64)
acc65_correct, acc65_total, acc65_pct = accuracy_stats(out_df, min_hit=64)

print("\n========== ACCURACY SUMMARY ==========")
print(f"Overall accuracy:          {overall_correct}/{overall_total} ({overall_pct:.1f}%)")
print(f"Accuracy (hit ≤ 16):       {acc16_correct}/{acc16_total} ({acc16_pct:.1f}%)")
print(f"Accuracy (hit ≤ 32):       {acc32_correct}/{acc32_total} ({acc32_pct:.1f}%)")
print(f"Accuracy (hit ≤ 64):       {acc64_correct}/{acc64_total} ({acc64_pct:.1f}%)")
print(f"Accuracy (hit > 64):       {acc65_correct}/{acc65_total} ({acc65_pct:.1f}%)")

# ——————————————
# Save output
# ——————————————
out_df.to_csv(save_path, index=False)
print(f"\n[INFO] Results saved to: {save_path}")
