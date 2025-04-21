import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse


model_path = "Qwen/Qwen2.5-3B"
csv_path = "qwen_math_eval_A1.csv" 
##############################################
# Prompt Template
##############################################
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)

def build_first_turn_prompt(problem: str) -> str:
    return SYSTEM_PROMPT.format(prompt=problem)

##############################################
# Reward and Utility Functions
##############################################
def reward_func(s, gt):
    extract_boxed_answer = last_boxed_only_string(s)
    if extract_boxed_answer is None: 
        return 0   
    return 1 if verify(parse(extract_boxed_answer), parse(gt)) else 0

def format_agnostic_reward(response: str, ground_truth: str) -> float:
    return 1.0 if verify(parse(response), parse(ground_truth)) else 0.0

def extract_ground_truth(solution: str) -> str:
    return last_boxed_only_string(solution)

##############################################
# Data and Model Setup
##############################################
def get_math_test_data():
    return load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]


llm = LLM(model=model_path)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=2500
)

##############################################
# Stats and Logging
##############################################
def print_turn_stats(df, turn_label="A1"):
    print(f"\nAverage Metrics for {turn_label}:")
    avg_reward = df[f"{turn_label}_total_reward"].mean()
    avg_length = df[f"{turn_label}_token_length"].mean()
    print(f"  Total Reward: {avg_reward}")
    print(f"  Token Length: {avg_length}")

    total = len(df)
    correct = (df[f"{turn_label}_total_reward"] == 1).sum()
    print(f"  Correct: {correct} ({(correct/total)*100:.2f}%)")
    print(f"  Incorrect: {total - correct} ({((total - correct)/total)*100:.2f}%)")

    if f"{turn_label}_format_agnostic" in df.columns:
        fa_counts = {val: (df[f"{turn_label}_format_agnostic"] == val).sum() for val in [1, 0]}
        print(f"  Format-Agnostic -> 1: {fa_counts[1]}, 0: {fa_counts[0]}")

##############################################
# Main Evaluation
##############################################
def main():
    print("Loading test data...")
    data = get_math_test_data()

    records = []
    prompts1 = []
    for ex in data:
        prompt1 = build_first_turn_prompt(ex["problem"])
        gt = extract_ground_truth(ex["solution"])
        records.append({"problem": ex["problem"], "ground_truth": gt})
        prompts1.append(prompt1)

    print("Generating A1...")
    outs1 = llm.generate(prompts1, sampling_params)
    for rec, out in zip(records, outs1):
        a1 = out.outputs[0].text
        rec.update({
            "A1": a1,
            "A1_total_reward": reward_func(a1, rec["ground_truth"]),
            "A1_format_agnostic": format_agnostic_reward(a1, rec["ground_truth"]),
            "A1_token_length": len(llm.get_tokenizer().encode(a1))
        })

    df1 = pd.DataFrame(records)
    print_turn_stats(df1, "A1")
    df1.to_csv(csv_path, index=False)
    print("Saved results to qwen_math_eval_A1.csv")

if __name__ == "__main__":
    main()
