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
# parser.add_argument("--csv_path", type=str, default="MATH_functional.csv")
parser.add_argument("--csv_path", type=str, default="MATH_perturb.csv")
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
# Reward
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
num_questions = len(df)

# Prepare variants: (prompt, ground_truth)
variants = {
    # "modified": {
    #     "prompts": df["problem"].tolist(),
    #     "ground_truths": [last_boxed_only_string(s) for s in df["solution"].tolist()],
    # },
    # "original": {
    #     "prompts": df["original_problem"].tolist(),
    #     "ground_truths": [last_boxed_only_string(s) for s in df["original_solution"].tolist()],
    # }
    "simple": {
        "prompts": df["problem_simple"].tolist(),
        "ground_truths": [last_boxed_only_string(s) for s in df["answer_simple"].tolist()],
    },
    "hard": {
        "prompts": df["problem_hard"].tolist(),
        "ground_truths": [last_boxed_only_string(s) for s in df["answer_hard"].tolist()],
    }
    
}
question_indices = df["question_index"].tolist()
hits = df["Unnamed: 0"].tolist()  # replace with df["hit"].tolist() if needed

# ———————————————————
# Run LLM inference for each variant
# ———————————————————
llm = LLM(model=model_path, max_model_len=4000, tensor_parallel_size=1)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_tokens=4000,
    n=n,
)

for variant, data in variants.items():
    prompts = data["prompts"]
    ground_truths = data["ground_truths"]

    print(f"[INFO] Sampling {n} completions for {variant} ({n * num_questions} generations)...")
    outputs = llm.generate(prompts, sampling_params)

    responses_matrix = np.empty((n, num_questions), dtype=object)
    rewards_matrix = np.zeros((n, num_questions), dtype=int)

    for i, (gt, out) in enumerate(zip(ground_truths, outputs)):
        for j in range(n):
            resp = out.outputs[j].text
            responses_matrix[j, i] = resp
            rewards_matrix[j, i] = reward_without_format(resp, gt)

    # Accuracy stats
    trial_means = rewards_matrix.mean(axis=1)
    mean_accuracy = trial_means.mean()
    std_accuracy = trial_means.std()

    print(f"\n========== ACCURACY ({variant}) ==========")
    print(f"Trial accuracies:      {trial_means}")
    print(f"Mean of accuracies:    {mean_accuracy:.4f}")
    print(f"Std dev of accuracies: {std_accuracy:.6f}")

    # Save long-form results
    records = []
    for i in range(n):
        for j in range(num_questions):
            records.append({
                "variant": variant,
                "trial_index": i,
                "question_index": question_indices[j],
                "hit": hits[j],
                "prompt": prompts[j],
                "ground_truth": ground_truths[j],
                "response": responses_matrix[i, j],
                "reward": rewards_matrix[i, j],
            })
    df_out = pd.DataFrame(records)
    df_out.to_csv(save_path.replace(".csv", f"_{variant}.csv"), index=False)
    print(f"[INFO] Saved results to {save_path.replace('.csv', f'_{variant}.csv')}")
