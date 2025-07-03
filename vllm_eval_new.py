import re
import pandas as pd
import numpy as np
import argparse
from vllm import LLM, SamplingParams
from math_verify import verify, parse

# ———————————————————
# Argument Parsing
# ———————————————————
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--csv_path", type=str, default="math_base_model_test_question_solution_hit.csv")
parser.add_argument("--n", type=int, default=15, help="Number of samples per prompt")
args = parser.parse_args()

model_path = args.model_path
save_path = args.save_path
csv_path = args.csv_path
n = args.n

temperature = 0.9
top_p = 1
top_k = 50

# ———————————————————
# Helper: Extract \boxed{} expression
# ———————————————————
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

# ———————————————————
# Reward (no formatting constraint)
# ———————————————————
def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

# ———————————————————
# Load dataset
# ———————————————————
df = pd.read_csv(csv_path)
prompts = df["prompt"].tolist()
ground_truths = [last_boxed_only_string(gt) for gt in df["ground_truth"].tolist()]
question_indices = df["question_index"].tolist()
hits = df["hit"].tolist()
num_questions = len(prompts)

# ———————————————————
# Run LLM inference
# ———————————————————
print(f"[INFO] Sampling {n} completions for {num_questions} prompts ({n * num_questions} total generations)...")
llm = LLM(model=model_path, max_model_len=4000, tensor_parallel_size=1)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_tokens=4000,
    n=n,
)

outputs = llm.generate(prompts, sampling_params)

# ———————————————————
# Process outputs into [n, num_questions]
# ———————————————————
responses_matrix = np.empty((n, num_questions), dtype=object)
rewards_matrix = np.zeros((n, num_questions), dtype=int)

for i, (gt, out) in enumerate(zip(ground_truths, outputs)):
    for j in range(n):
        resp = out.outputs[j].text
        responses_matrix[j, i] = resp
        rewards_matrix[j, i] = reward_without_format(resp, gt)

# ———————————————————
# Accuracy Stats
# ———————————————————
trial_means = rewards_matrix.mean(axis=1)
mean_accuracy = trial_means.mean()
std_accuracy = trial_means.std()

print("\n========== OVERALL ACCURACY ==========")
print(f"Trial accuracies:            {trial_means}")
print(f"Mean of accuracies:          {mean_accuracy:.4f}")
print(f"Std dev of accuracies:       {std_accuracy:.6f}")

# Accuracy by hit bucket for sample 0
def accuracy_stats(bucket_rewards, bucket_hits, min_hit=None, max_hit=None):
    mask = np.ones(len(bucket_hits), dtype=bool)
    if min_hit is not None:
        mask &= bucket_hits > min_hit
    if max_hit is not None:
        mask &= bucket_hits <= max_hit
    total = mask.sum()
    correct = bucket_rewards[mask].sum()
    percent = (correct / total * 100) if total > 0 else 0.0
    return correct, total, percent

sample0_rewards = rewards_matrix[0]
hits_array = np.array(hits)

print("\n========== FIRST SAMPLE BUCKET ACCURACY ==========")
for name, min_hit, max_hit in [
    ("hit ≤ 16", None, 16),
    ("hit ≤ 32", None, 32),
    ("hit ≤ 64", None, 64),
    ("hit > 64", 64, None),
]:
    correct, total, pct = accuracy_stats(sample0_rewards, hits_array, min_hit, max_hit)
    print(f"{name.ljust(20)} {correct}/{total} ({pct:.1f}%)")

# ———————————————————
# Save all responses in long-form
# ———————————————————
records = []
for i in range(n):
    for j in range(num_questions):
        records.append({
            "trial_index": i,
            "question_index": question_indices[j],
            "hit": hits[j],
            "prompt": prompts[j],
            "ground_truth": ground_truths[j],
            "response": responses_matrix[i, j],
            "reward": rewards_matrix[i, j]
        })

df_out = pd.DataFrame(records)
df_out.to_csv(save_path, index=False)
print(f"\n[INFO] Saved all {n}×{num_questions} completions to {save_path}")
