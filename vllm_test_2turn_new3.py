import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse

# ——————————————
# Prompt Templates
# ——————————————
PROMPT_TEMPLATES = {
#     "TYPE1": """
# <|im_start|>system
# Please reason step by step, and put your final answer within \\boxed{{}}.
# <|im_end|>
# <|im_start|>user
# {prompt}
# <|im_end|>
# <|im_start|>assistant
# """,
    "TYPE2": """
<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
""",
    # "TYPE3": "{prompt}"
}

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
model_path = "Qwen/Qwen2.5-3B"
llm = LLM(model=model_path)
sampling_params = SamplingParams(temperature=0, max_tokens=2560, top_p=0.9)

# ——————————————
# Main loop over prompt types
# ——————————————
for prompt_type, SYSTEM_PROMPT in PROMPT_TEMPLATES.items():
    print(f"Running inference with {prompt_type}...")

    # Prepare prompts and records
    records = []
    prompts = []
    for ex in test_ds:
        prompt = SYSTEM_PROMPT.format(prompt=ex["problem"])
        gt = last_boxed_only_string(ex["solution"])
        records.append({"problem": ex["problem"], "prompt": prompt, "ground_truth": gt})
        prompts.append(prompt)

    # Generate responses
    outs = llm.generate(prompts, sampling_params)

    # Evaluate responses
    results = []
    for rec, out in zip(records, outs):
        a1 = out.outputs[0].text
        rec["response"] = a1
        gt = rec["ground_truth"]
        rec["reward_with_format"] = reward_with_format(a1, gt)
        rec["reward_without_format"] = reward_without_format(a1, gt)
        results.append(rec)

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = f"qwen_3b_base_eval2_{prompt_type}.csv"
    df.to_csv(csv_path, index=False)

    # Print summary stats
    mean_with_format = df["reward_with_format"].mean()
    mean_without_format = df["reward_without_format"].mean()

    print(f"[{prompt_type}] Mean Reward (with format): {mean_with_format:.3f}")
    print(f"[{prompt_type}] Mean Reward (without format): {mean_without_format:.3f}\n")

    def print_reward_counts(col_name):
        counts = df[col_name].value_counts().sort_index()
        print(f"[{prompt_type}] {col_name} distribution:\n{counts.to_string()}\n")

    print_reward_counts("reward_with_format")
    print_reward_counts("reward_without_format")
