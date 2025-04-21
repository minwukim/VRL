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
csv_path   = "qwen_3b_base.csv"

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
# Reward functions (you already wrote these)
# ——————————————
def reward_with_format(s, gt):
    answer = last_boxed_only_string(s)
    if answer is None:
        return 0
    return int(verify(parse(answer), parse(gt)))

def reward_without_format(s, gt):
    return int(verify(parse(s), parse(gt)))

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
for sample in test_ds:
    question = sample["question"]      # adjust key if your split uses a different field
    ground_truth = sample["answer"]    # adjust if necessary

    # build the full prompt
    prompt = SYSTEM_PROMPT.format(prompt=question)

    # generate one response
    outputs = llm.generate([
        {"prompt": prompt, "sampling_params": sampling_params}
    ])
    # grab the text of the very first output
    response_text = next(outputs).outputs[0].text

    # compute rewards
    r_fmt = reward_with_format(response_text, ground_truth)
    r_nofmt = reward_without_format(response_text, ground_truth)

    # collect
    records.append({
        "question":           question,
        "response":           response_text,
        "ground_truth":       ground_truth,
        "reward_with_format": r_fmt,
        "reward_no_format":   r_nofmt
    })

# ——————————————
# Save + report
# ——————————————
df = pd.DataFrame.from_records(records)
df.to_csv(csv_path, index=False)

mean_with   = df["reward_with_format"].mean()
mean_no_fmt = df["reward_no_format"].mean()

print(f"Saved results to {csv_path}")
print(f"Mean reward (with format):    {mean_with:.4f}")
print(f"Mean reward (without format): {mean_no_fmt:.4f}")
