import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Global Prompt Templates
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

CORRECTION_INSTRUCTION = (
    "<|im_end|>\n<|im_start|>user\n"
    "There might be an error in the solution above because of lack of understanding of the question. "
    "Please correct the error, if any, and rewrite the solution. Maintain the format of: "
    "<think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>."
    "<|im_end|>\n<|im_start|>assistant\n<think>"
)

##############################################
# Prompt Building Functions
##############################################

def build_first_turn_prompt(problem: str) -> str:
    """
    Build the first-turn prompt using the SYSTEM_PROMPT template.
    """
    return SYSTEM_PROMPT.format(prompt=problem)

def build_second_turn_prompt(problem: str, a1_response: str) -> str:
    """
    Build the second-turn prompt using the exact same structure as in training:
    {prompt}{completion}{instruction}.
    
    This helps ensure that the model receives an input with an expected format.
    """
    first_prompt = build_first_turn_prompt(problem)
    # Concatenate the first-turn prompt, the A1 response, and the correction instruction.
    return f"{first_prompt}{a1_response}{CORRECTION_INSTRUCTION}"

##############################################
# Reward and Utility Functions
##############################################

def reward_func(response: str, ground_truth: str) -> float:
    """
    Computes a reward for a given response string based on the ground truth.
    It verifies that the response is correctly formatted (exactly one </think>, one <answer>, and one </answer>)
    and then checks for a boxed answer.
    
    Returns:
      2   if the format is correct and the boxed answer matches the ground truth,
      -0.5 if the format is correct but the answer is incorrect,
      -1   if the boxed answer is missing, or
      -2   if the overall format is wrong.
    """
    pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
    if not (response.count("</think>") == 1 and response.count("<answer>") == 1 and response.count("</answer>") == 1):
        return -2
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return -2
    extracted = last_boxed_only_string(match.group(1))
    if extracted is None:
        return -1
    return 2 if verify(parse(extracted), parse(ground_truth)) else -0.5

def format_agnostic_reward(response: str, ground_truth: str) -> float:
    """
    Computes a format-agnostic reward.
    Returns 1.0 if the parsed answer matches the ground truth regardless
    of extra formatting tokens, else returns 0.0.
    """
    return 1.0 if verify(parse(response), parse(ground_truth)) else 0.0

def extract_ground_truth(solution: str) -> str:
    """
    Extract the last boxed answer from the reference solution.
    """
    return last_boxed_only_string(solution)

##############################################
# Data and Model Setup
##############################################

def get_math_test_data():
    """
    Loads the MATH-500 test dataset.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]
    return dataset

# Set your model checkpoint path
model_path = "./0415-qwen3b-it-ONN/checkpoint-300"  # adjust if necessary
llm = LLM(model=model_path)

# Define sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=2500
)

##############################################
# Statistics and Logging Functions
##############################################

def print_turn_stats(df, turn_label="A1"):
    print(f"\nAverage Metrics for {turn_label}:")
    avg_reward = df[f"{turn_label}_total_reward"].mean()
    avg_length = df[f"{turn_label}_token_length"].mean()
    print(f"  Total Reward: {avg_reward}")
    print(f"  Token Length: {avg_length}")

    total = df.shape[0]
    correct = (df[f"{turn_label}_total_reward"] == 2).sum()
    print(f"  Correct: {correct} ({(correct/total)*100:.2f}%)")
    print(f"  Incorrect: {total - correct} ({((total - correct)/total)*100:.2f}%)")

    count_2 = (df[f"{turn_label}_total_reward"] == 2).sum()
    count_m05 = (df[f"{turn_label}_total_reward"] == -0.5).sum()
    count_m1 = (df[f"{turn_label}_total_reward"] == -1).sum()
    count_m2 = (df[f"{turn_label}_total_reward"] == -2).sum()
    print(f"  Counts -> 2: {count_2}, -0.5: {count_m05}, -1: {count_m1}, -2: {count_m2}")

    if f"{turn_label}_format_agnostic" in df.columns:
        count_1 = (df[f"{turn_label}_format_agnostic"] == 1).sum()
        count_0 = (df[f"{turn_label}_format_agnostic"] == 0).sum()
        print(f"  Format-Agnostic -> 1: {count_1}, 0: {count_0}")

def print_confusion_matrix(df):
    # Confusion matrix based on reward_func scores
    c_to_c = ((df["A1_total_reward"] == 2) & (df["A2_total_reward"] == 2)).sum()
    c_to_i = ((df["A1_total_reward"] == 2) & (df["A2_total_reward"] != 2)).sum()
    i_to_c = ((df["A1_total_reward"] != 2) & (df["A2_total_reward"] == 2)).sum()
    i_to_i = ((df["A1_total_reward"] != 2) & (df["A2_total_reward"] != 2)).sum()
    print("\nConfusion Matrix (Reward Function Scores, A1 -> A2):")
    print(f"  Correct to Correct: {c_to_c}")
    print(f"  Correct to Incorrect: {c_to_i}")
    print(f"  Incorrect to Correct: {i_to_c}")
    print(f"  Incorrect to Incorrect: {i_to_i}")

    # Confusion matrix for format-agnostic scores
    fa_c_to_c = ((df["A1_format_agnostic"] == 1) & (df["A2_format_agnostic"] == 1)).sum()
    fa_c_to_i = ((df["A1_format_agnostic"] == 1) & (df["A2_format_agnostic"] == 0)).sum()
    fa_i_to_c = ((df["A1_format_agnostic"] == 0) & (df["A2_format_agnostic"] == 1)).sum()
    fa_i_to_i = ((df["A1_format_agnostic"] == 0) & (df["A2_format_agnostic"] == 0)).sum()
    print("\nConfusion Matrix (Format-Agnostic, A1 -> A2):")
    print(f"  Correct to Correct: {fa_c_to_c}")
    print(f"  Correct to Incorrect: {fa_c_to_i}")
    print(f"  Incorrect to Correct: {fa_i_to_c}")
    print(f"  Incorrect to Incorrect: {fa_i_to_i}")

##############################################
# Main Two-Turn Evaluation Process
##############################################

def main():
    print("Loading test data...")
    data = get_math_test_data()
    
    # --- FIRST TURN ---
    prompts_first = []
    ground_truths = []
    for example in data:
        prompt_first = build_first_turn_prompt(example["problem"])
        prompts_first.append(prompt_first)
        gt = extract_ground_truth(example["solution"])
        ground_truths.append(gt)

    print("\nRunning First Turn Generation...")
    print(prompts_first[0])
    outputs_turn1 = llm.generate(prompts_first, sampling_params)

    first_turn_data = []
    for idx, out in enumerate(outputs_turn1):
        a1_response = out.outputs[0].text
        gt = ground_truths[idx]
        reward_a1 = reward_func(a1_response, gt)
        fa_a1 = format_agnostic_reward(a1_response, gt)
        token_length_a1 = len(llm.get_tokenizer().encode(a1_response))
        
        first_turn_data.append({
            "problem": data[idx]["problem"],
            "prompt_first": prompts_first[idx],
            "ground_truth": gt,
            "A1": a1_response,
            "A1_total_reward": reward_a1,
            "A1_format_agnostic": fa_a1,
            "A1_token_length": token_length_a1,
        })

    df_first = pd.DataFrame(first_turn_data)
    print_turn_stats(df_first, turn_label="A1")

    # --- SECOND TURN ---
    prompts_second = []
    for item in first_turn_data:
        prompt_second = build_second_turn_prompt(item["problem"], item["A1"])
        prompts_second.append(prompt_second)
    
    print("\nRunning Second Turn Generation...")
    print(prompts_second[0])
    outputs_turn2 = llm.generate(prompts_second, sampling_params)

    final_data = []
    for idx, item in enumerate(first_turn_data):
        a2_response = outputs_turn2[idx].outputs[0].text
        gt = item["ground_truth"]
        reward_a2 = reward_func(a2_response, gt)
        fa_a2 = format_agnostic_reward(a2_response, gt)
        token_length_a2 = len(llm.get_tokenizer().encode(a2_response))
        
        merged_item = {
            **item,
            "prompt_second": prompts_second[idx],
            "A2": a2_response,
            "A2_total_reward": reward_a2,
            "A2_format_agnostic": fa_a2,
            "A2_token_length": token_length_a2,
        }
        final_data.append(merged_item)
    
    df_final = pd.DataFrame(final_data)
    print_turn_stats(df_final, turn_label="A2")
    print_confusion_matrix(df_final)
    
    # # Save results to CSV.
    # csv_file_path = "evaluation_results.csv"
    # df_final.to_csv(csv_file_path, index=False)
    # print(f"\nEvaluation complete. Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()
