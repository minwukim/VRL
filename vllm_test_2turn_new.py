import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Settings and Model Initialization
##############################################
# model_path = "./outputs/qwen2.5-3b-grpo-full/checkpoint-400"  # or update to your desired model path
# model_path = "./qwen2.5-3b-grpo-switch/checkpoint-350"  # or update to your desired model path
# model_path = "./qwen2.5-3b-grpo-switch-type1-reward/checkpoint-300"
# model_path = "Qwen/Qwen2.5-3B-instruct"
# model_path = "./qwen3b-it-old-prompt/checkpoint-450"
model_path = "./0415-qwen3b-it-ONN-a1agnostic/checkpoint-250"
# model_path = "./0414-qwen3b-it-ONN/checkpoint-150"
# model_path = "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero"
# csv_file_path = "./qwen2.5-grpo-switch-csvs/qwen2.5-3b-grpo-switch-type1-reward-checkpoint-350_2stage.csv"


# First turn prompt template
SYSTEM_PROMPT_FIRST="""
<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>.<|im_end|>\n<|im_start|>user\n{prompt}.<|im_end|>\n<|im_start|>assistant\n<think>
"""

# # Instruction for the second turn (correction/verification)
# ADDED_INSTRUCTION = (
#     "A conversation between User and Assistant. Given a question and a corresponding response provided below, the Assistant systematically reviews and explains each step of the reasoning process to verify the correctness of the response."
#     "If errors are found, the Assistant identifies and corrects them, then re-solves the problem. If the response is correct, the Assistant confirms it and returns the same final answer."
#     "The assistant first thinks about the reasoning process in mind, including verification, correction, and resolving the problem if necessary. Then provides the user with the answer."
#     "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>." 
#     "The reasoning process, including verification and correction, is enclosed within <think> </think> tags, while the final solution is enclosed within <answer> </answer> tags. The final answer is formatted within \\boxed{{}} to enable direct extraction for grading."
#     "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag."
# )


# ADDED_INSTRUCTION = "User: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Maintain the format of: <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>. \nAssistant: <think>"
ADDED_INSTRUCTION = "<|im_end|>\n<|im_start|>user\n There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Maintain the format of: <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>.<|im_end|>\n<|im_start|>assistant\n <think>"
# Second turn prompt template
SYSTEM_PROMPT_SECOND = (
    "{instruction}"
    + "\nQuestion:\n{question}"
    + "\nResponse:\n<think>{first_answer}"
    + "\nAssistant: <think>" 
)


##############################################
# Prompt and Utility Functions
##############################################
def build_prompt_first(problem: str) -> str:
    """Build the first-turn prompt by inserting the problem into the system prompt."""
    return SYSTEM_PROMPT_FIRST.format(prompt=problem)


def build_prompt_second(question: str, first_answer: str) -> str:
    """Build the second-turn prompt using the original problem and the first-turn answer."""
    return SYSTEM_PROMPT_SECOND.format(instruction=ADDED_INSTRUCTION, question=question, first_answer=first_answer)


def reward_func(s: str, gt: str) -> float:
    """
    Computes a reward for the given answer string `s` using the ground truth `gt`.
    Returns 2 if the answer is correctly formatted and the extracted final answer matches the ground truth.
    Otherwise returns other values (e.g., -2, -1, or -0.5) indicating error cases.
    """
    pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
    # Check that the answer has the proper number of tokens.
    if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
        return -2
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        return -2
    # Answer format is correct; now extract the answer inside the boxed tag.
    ext_string = last_boxed_only_string(match.group(1))
    if ext_string is None:
        return -1   # No boxed tag found
    # Return 2 if the parsed answer matches the ground truth, else return -0.5.
    return 2 if verify(parse(ext_string), parse(gt)) else -0.5


def format_agnostic_reward_func(completion: str, ground_truth: str) -> float:
    """
    Directly computes the correctness regardless of formatting.
    Returns 1.0 if the parsed completion matches the ground truth; otherwise 0.0.
    """
    return 1.0 if verify(parse(completion), parse(ground_truth)) else 0.0


def extract_ground_truth(text: str) -> str:
    """Extract the last boxed answer from the reference solution."""
    return last_boxed_only_string(text)


def get_math_test_data():
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)["test"]
    return data

##############################################
# LLM and Sampling Parameters
##############################################
llm = LLM(model=model_path)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=1.0,
    max_tokens=6000
)

##############################################
# Statistics and Confusion Matrix Helpers
##############################################
def print_turn_stats(df, turn_label="A1"):
    print(f"\nAverage Metrics for {turn_label}:")
    avg_reward = df[f"{turn_label}_total_reward"].mean()
    avg_length = df[f"{turn_label}_token_length"].mean()

    print(f"  Total Reward: {avg_reward}")
    print(f"  Token Length: {avg_length}")

    total_examples = df.shape[0]
    correct_count = (df[f"{turn_label}_total_reward"] == 2).sum()
    incorrect_count = total_examples - correct_count

    # Count individual reward values from reward_func
    count_2   = (df[f"{turn_label}_total_reward"] == 2).sum()
    count_m05 = (df[f"{turn_label}_total_reward"] == -0.5).sum()
    count_m1  = (df[f"{turn_label}_total_reward"] == -1).sum()
    count_m2  = (df[f"{turn_label}_total_reward"] == -2).sum()

    print(f"\n{turn_label} Reward (using reward_func) Accuracy:")
    print(f"  Correct: {correct_count} ({(correct_count / total_examples)*100:.2f}%)")
    print(f"  Incorrect: {incorrect_count} ({(incorrect_count / total_examples)*100:.2f}%)")
    print(f"\n{turn_label} Reward (using reward_func) Counts:")
    print(f"  2     : {count_2}")
    print(f"  -0.5  : {count_m05}")
    print(f"  -1    : {count_m1}")
    print(f"  -2    : {count_m2}")
    
    # If format-agnostic scores are available, print their counts.
    if f"{turn_label}_format_agnostic" in df.columns:
        count_1 = (df[f"{turn_label}_format_agnostic"] == 1).sum()
        count_0 = (df[f"{turn_label}_format_agnostic"] == 0).sum()
        print(f"\n{turn_label} Format-Agnostic Reward Counts:")
        print(f"  1 : {count_1}")
        print(f"  0 : {count_0}")


def print_confusion_matrix(df):
    # Confusion matrix for reward_func-based scores (correct = 2)
    correct_to_correct = ((df["A1_total_reward"] == 2) & (df["A2_total_reward"] == 2)).sum()
    correct_to_incorrect = ((df["A1_total_reward"] == 2) & (df["A2_total_reward"] != 2)).sum()
    incorrect_to_correct = ((df["A1_total_reward"] != 2) & (df["A2_total_reward"] == 2)).sum()
    incorrect_to_incorrect = ((df["A1_total_reward"] != 2) & (df["A2_total_reward"] != 2)).sum()

    print("\nConfusion Matrix (reward_func scores, A1 -> A2):")
    print(f"  Correct to Correct: {correct_to_correct}")
    print(f"  Correct to Incorrect: {correct_to_incorrect}")
    print(f"  Incorrect to Correct: {incorrect_to_correct}")
    print(f"  Incorrect to Incorrect: {incorrect_to_incorrect}")

    # Confusion matrix for format-agnostic scores (correct = 1)
    fa_correct_to_correct = ((df["A1_format_agnostic"] == 1) & (df["A2_format_agnostic"] == 1)).sum()
    fa_correct_to_incorrect = ((df["A1_format_agnostic"] == 1) & (df["A2_format_agnostic"] == 0)).sum()
    fa_incorrect_to_correct = ((df["A1_format_agnostic"] == 0) & (df["A2_format_agnostic"] == 1)).sum()
    fa_incorrect_to_incorrect = ((df["A1_format_agnostic"] == 0) & (df["A2_format_agnostic"] == 0)).sum()

    print("\nConfusion Matrix (Format-Agnostic scores, A1 -> A2):")
    print(f"  Correct to Correct: {fa_correct_to_correct}")
    print(f"  Correct to Incorrect: {fa_correct_to_incorrect}")
    print(f"  Incorrect to Correct: {fa_incorrect_to_correct}")
    print(f"  Incorrect to Incorrect: {fa_incorrect_to_incorrect}")

##############################################
# Main Two-Stage Process
##############################################
def main():
    print("Loading dataset...")
    data = get_math_test_data()

    # --- FIRST TURN ---
    prompts_first = []
    ground_truths = []
    for example in data:
        prompt_first = build_prompt_first(example["problem"])
        gt = extract_ground_truth(example["solution"])
        prompts_first.append(prompt_first)
        ground_truths.append(gt)

    print("\nRunning FIRST TURN for all examples...")
    print(prompts_first[0])
    outputs_turn1 = llm.generate(prompts_first, sampling_params)

    partial_data = []
    for idx, out in enumerate(outputs_turn1):
        A1 = out.outputs[0].text
        gt = ground_truths[idx]
        score_A1 = reward_func(A1, gt)
        fa_A1 = format_agnostic_reward_func(A1, gt)
        total_reward_A1 = score_A1
        token_length_A1 = len(llm.get_tokenizer().encode(A1))

        partial_data.append({
            "problem": data[idx]["problem"],
            "prompt_first": prompts_first[idx],
            "ground_truth": gt,
            "A1": A1,
            "A1_total_reward": total_reward_A1,
            "A1_format_agnostic": fa_A1,
            "A1_token_length": token_length_A1,
        })

    df_tmp = pd.DataFrame(partial_data)
    print_turn_stats(df_tmp, turn_label="A1")

    # --- SECOND TURN ---
    prompts_second = []
    # for item in partial_data:
    #     prompt_second = build_prompt_second(item["problem"], item["A1"])
    #     prompts_second.append(prompt_second)

    for item in partial_data:
        # 1) Reconstruct the original system+question prompt
        q_text = SYSTEM_PROMPT_FIRST.format(prompt=item["problem"])
        # 2) Grab the first‐turn assistant response
        a1_text = item["A1"]
        # 3) Append your second‐turn instruction
        prompt_second = (
            q_text
            + a1_text
            + ADDED_INSTRUCTION
        )
        prompts_second.append(prompt_second)

    print("\nRunning SECOND TURN for all examples...")
    print(prompts_second[0])
    outputs_turn2 = llm.generate(prompts_second, sampling_params)

    final_data = []
    for idx, item in enumerate(partial_data):
        A2 = outputs_turn2[idx].outputs[0].text
        if idx==1:
            print(A2)
        gt = item["ground_truth"]
        score_A2 = reward_func(A2, gt)
        fa_A2 = format_agnostic_reward_func(A2, gt)
        total_reward_A2 = score_A2
        token_length_A2 = len(llm.get_tokenizer().encode(A2))

        merged = {
            **item,
            "prompt_second": prompts_second[idx],
            "A2": A2,
            "A2_total_reward": total_reward_A2,
            "A2_format_agnostic": fa_A2,
            "A2_token_length": token_length_A2,
        }
        final_data.append(merged)

    df_final = pd.DataFrame(final_data)
    print_turn_stats(df_final, turn_label="A2")
    print_confusion_matrix(df_final)

    # df_final.to_csv(csv_file_path, index=False)
    # print(f"\nAll done. Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()

