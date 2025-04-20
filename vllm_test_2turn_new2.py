import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Prompt Templates
##############################################
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning process "
    "in mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think> </think> and <answer> \\boxed{{final answer inside}} </answer> tags, "
    "respectively.<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>"
)

SYSTEM_PROMPT="""
<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
"""

CONFIDENCE_PROMPT = (
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Rate your confidence in the preceding response on a scale of 0 (likely incorrect) to 10 (likely correct), inclusive. "
    "Output only your integer confidence score, with the format `<confidence>score</confidence>`.\n"
    "<|im_end|>\n"
)

REVIEW_PROMPT = (
    "<|im_start|>user\n"
    "Review your previous response considering your confidence score, and if needed, correct any errors. "
    "Provide your revised solution in the format: `<think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>`.\n"
    "<|im_end|>\n"
)

##############################################
# Prompt Builders
##############################################
def build_first_turn_prompt(problem: str) -> str:
    return SYSTEM_PROMPT.format(prompt=problem)


def build_second_turn_prompt(problem: str, a1_response: str, fa_score: float) -> str:
    """
    Build the second-turn prompt by appending the first response, a confidence step, and a review instruction.
    If format-agnostic score == 1.0, the model is fully confident (10); otherwise confidence=0.
    """
    first_prompt = build_first_turn_prompt(problem)
    confidence_value = 10 if fa_score >= 1.0 else 0

    return (
        f"{first_prompt}{a1_response}"
        f"{CONFIDENCE_PROMPT}"
        f"<|im_start|>assistant\n<confidence>{confidence_value}</confidence><|im_end|>\n"
        f"{REVIEW_PROMPT}"
        f"<|im_start|>assistant\n<think>"
    )

##############################################
# Reward and Utility Functions
##############################################
# def reward_func(response: str, ground_truth: str) -> float:
#     pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
#     if not (response.count("</think>") == 1 and response.count("<answer>") == 1 and response.count("</answer>") == 1):
#         return -2
#     match = re.search(pattern, response, re.DOTALL)
#     if not match:
#         return -2
#     extracted = last_boxed_only_string(match.group(1))
#     if extracted is None:
#         return -1
#     return 2 if verify(parse(extracted), parse(ground_truth)) else -0.5

def reward_func(s, gt):
    extract_boxed_answer = last_boxed_only_string(s)
    if extract_boxed_answer is None: 
        return 0   
    
    if verify(parse(extract_boxed_answer), parse(gt)): 
        return 1
    return 0



def format_agnostic_reward(response: str, ground_truth: str) -> float:
    return 1.0 if verify(parse(response), parse(ground_truth)) else 0.0


def extract_ground_truth(solution: str) -> str:
    return last_boxed_only_string(solution)

##############################################
# Data and Model Setup
##############################################
def get_math_test_data():
    
    return load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]
    # return load_dataset("DigitalLearningGmbH/MATH-lighteval", trust_remote_code=True)["test"]

# model_path = "./0417-qwen3b-it-OON-oracle-switch/checkpoint-200"
# model_path = "./qwen3b-it-old-prompt/checkpoint-350"
model_path = "./qwen3b-it-SFT-boxed/checkpoint-175"
# model_path = "Qwen/Qwen2.5-3B"
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

    # counts = {score: (df[f"{turn_label}_total_reward"] == score).sum() for score in [2, -0.5, -1, -2]}
    # print(f"  Counts -> 2: {counts[2]}, -0.5: {counts[-0.5]}, -1: {counts[-1]}, -2: {counts[-2]}")

    if f"{turn_label}_format_agnostic" in df.columns:
        fa_counts = {val: (df[f"{turn_label}_format_agnostic"] == val).sum() for val in [1, 0]}
        print(f"  Format-Agnostic -> 1: {fa_counts[1]}, 0: {fa_counts[0]}")


def print_confusion_matrix(df):
    c_to_c = ((df["A1_format_agnostic"] == 1) & (df["A2_format_agnostic"] == 1)).sum()
    c_to_i = ((df["A1_format_agnostic"] == 1) & (df["A2_format_agnostic"] == 0)).sum()
    i_to_c = ((df["A1_format_agnostic"] == 0) & (df["A2_format_agnostic"] == 1)).sum()
    i_to_i = ((df["A1_format_agnostic"] == 0) & (df["A2_format_agnostic"] == 0)).sum()
    print("\nConfusion Matrix (Format-Agnostic Correctness, A1 -> A2):")
    print(f"  Correct to Correct: {c_to_c}")
    print(f"  Correct to Incorrect: {c_to_i}")
    print(f"  Incorrect to Correct: {i_to_c}")
    print(f"  Incorrect to Incorrect: {i_to_i}")

##############################################
# Main Evaluation
##############################################
def main():
    print("Loading test data...")
    data = get_math_test_data()

    # FIRST TURN
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

    # SECOND TURN using format-agnostic correctness
    prompts2 = [
        build_second_turn_prompt(r["problem"], r["A1"], r["A1_format_agnostic"])
        for r in records
    ]

    print("Generating A2...")
    outs2 = llm.generate(prompts2, sampling_params)
    for rec, out in zip(records, outs2):
        a2 = out.outputs[0].text
        rec.update({
            "A2": a2,
            "A2_total_reward": reward_func(a2, rec["ground_truth"]),
            "A2_format_agnostic": format_agnostic_reward(a2, rec["ground_truth"]),
            "A2_token_length": len(llm.get_tokenizer().encode(a2))
        })

    df_final = pd.DataFrame(records)
    print_turn_stats(df_final, "A2")
    print_confusion_matrix(df_final)

if __name__ == "__main__":
    main()
