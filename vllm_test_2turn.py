import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Single-file script that performs two-stage evaluation.
# 1) First turn: generates an answer for each item.
# 2) Second turn: re-evaluates each item using the first turn's output.
# 3) Exports a single CSV file containing the results for both turns.
# 4) Prints out the same statistics as in the previous code.
##############################################

# Choose your checkpoint path or model name
path = "Qwen/Qwen2.5-3B-Instruct"

# The path name for the final CSV file to save the combined results.
csv_file_path = "2stage_qwen2.5_3b_it_original.csv"

##############################################
# System Prompt and Additional Instruction
##############################################
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
    "Ensure that the final answer in the solution is formatted within \\boxed{}, "
    "as this formatting is required for correct extraction."
)

ADDED_INSTRUCTION = (
    "\n Given the question and the response provided, go through the reasoning process of the response and check if the response is correct or not. "
    "Then, try to resolve the problem if incorrect, and return the same final answer if you think it is correct. "
    "Enclose your reasoning of checking and resolving process within <think> </think> tags and the final solution within <answer> </answer> tags, "
    "i.e., <think> reasoning process here </think> <answer> solution here </answer>. "
    "Ensure that the final answer in the solution is formatted within \\boxed{}, as this formatting is required for correct extraction."
)

##############################################
# Helper Functions
##############################################
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def build_prompt_from_conversation(conversation):
    # Convert a conversation object to a single prompt string
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation["prompt"])

def extract_ground_truth(text: str) -> str | None:
    # Extract the last boxed answer from the reference solution.
    return last_boxed_only_string(text)

##############################################
# Reward Functions
##############################################
def correct_and_format(prompt: str, completion: str, ground_truth: str) -> float:
    # Extract the answer enclosed within <answer>...</answer>
    pattern = r"</think>\n?<answer>([\s\S]*)</answer>"
    match = re.search(pattern, completion)
    content = match.group(1) if match else ""
    return 1.0 if verify(parse(content), parse(ground_truth)) else 0.0

def format_reward_func(completion: str, ground_truth: str) -> float:
    # Checks if the entire completion matches the ground truth
    return 1.0 if verify(parse(completion), parse(ground_truth)) else 0.0

def boxed_format_reward_func(completion: str) -> float:
    # A small reward if there's at least one '\\boxed{}' in the text
    match = re.search(r"\\boxed\{(.*?)\}", completion)
    return 0.1 if match else 0.0

##############################################
# Load the MATH-500 Dataset
##############################################
def get_math_test_data():
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)["test"]
    return data

##############################################
# LLM Initialization
##############################################
llm = LLM(
    model=path
)

# Deterministic generation
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=4000
)

##############################################
# Statistics / Confusion Matrix Helpers
##############################################
def print_turn_stats(df, turn_label="A1"):
    print(f"\nAverage Metrics for {turn_label}:")
    avg_corr = df[f"{turn_label}_correctness"].mean()
    avg_format = df[f"{turn_label}_token_format"].mean()
    avg_boxed = df[f"{turn_label}_boxed_format"].mean()
    avg_reward = df[f"{turn_label}_total_reward"].mean()
    avg_length = df[f"{turn_label}_token_length"].mean()

    print(f"  Correctness: {avg_corr}")
    print(f"  Token Format: {avg_format}")
    print(f"  Boxed Format: {avg_boxed}")
    print(f"  Total Reward: {avg_reward}")
    print(f"  Token Length: {avg_length}")

    total_examples = len(df)
    correct_count = (df[f"{turn_label}_correctness"] == 1.0).sum()
    incorrect_count = (df[f"{turn_label}_correctness"] == 0.0).sum()

    print(f"\n{turn_label} Accuracy:")
    print(f"  Correct: {correct_count} ({(correct_count / total_examples)*100:.2f}%)")
    print(f"  Incorrect: {incorrect_count} ({(incorrect_count / total_examples)*100:.2f}%)")


def print_confusion_matrix(df):
    correct_to_correct = ((df["A1_correctness"] == 1.0) & (df["A2_correctness"] == 1.0)).sum()
    correct_to_incorrect = ((df["A1_correctness"] == 1.0) & (df["A2_correctness"] == 0.0)).sum()
    incorrect_to_correct = ((df["A1_correctness"] == 0.0) & (df["A2_correctness"] == 1.0)).sum()
    incorrect_to_incorrect = ((df["A1_correctness"] == 0.0) & (df["A2_correctness"] == 0.0)).sum()

    print("\nConfusion Matrix (A1 -> A2):")
    print(f"  Correct to Correct: {correct_to_correct}")
    print(f"  Correct to Incorrect: {correct_to_incorrect}")
    print(f"  Incorrect to Correct: {incorrect_to_correct}")
    print(f"  Incorrect to Incorrect: {incorrect_to_incorrect}")

##############################################
# Main Two-Stage Process
##############################################
def main():
    print("Loading dataset...")
    data = get_math_test_data()

    # Prepare empty list to store final results
    results = []

    print("\nRunning FIRST TURN for all examples...")
    # Build all conversations and gather ground truths
    conversations = []
    ground_truths = []
    for example in data:
        conv = make_conversation(example)
        gt = extract_ground_truth(example["solution"])
        conversations.append(conv)
        ground_truths.append(gt)

    # Build prompts for turn 1
    prompts_turn1 = [build_prompt_from_conversation(c) for c in conversations]

    # Generate outputs for turn 1
    outputs_turn1 = llm.generate(prompts_turn1, sampling_params)

    # Create a partial results list that we can fill in turn2
    partial_data = []
    for idx, out in enumerate(outputs_turn1):
        # Turn 1 answer
        A1 = out.outputs[0].text
        prompt = prompts_turn1[idx]
        gt = ground_truths[idx]

        score_corr_A1 = correct_and_format(prompt, A1, gt)
        score_format_A1 = format_reward_func(A1, gt)
        score_box_A1 = boxed_format_reward_func(A1)
        total_reward_A1 = score_corr_A1 + score_format_A1 + score_box_A1
        token_length_A1 = len(llm.get_tokenizer().encode(A1))

        partial_data.append({
            "question": prompt,
            "ground_truth": gt,
            "A1": A1,
            "A1_correctness": score_corr_A1,
            "A1_token_format": score_format_A1,
            "A1_boxed_format": score_box_A1,
            "A1_total_reward": total_reward_A1,
            "A1_token_length": token_length_A1,
        })

    # Convert partial_data to a DataFrame so we can do stats
    df_tmp = pd.DataFrame(partial_data)

    # Print stats for A1
    print_turn_stats(df_tmp, turn_label="A1")

    print("\nRunning SECOND TURN for all examples...")
    # Now build second turn prompts using the output of turn1
    second_prompts = []
    for idx, row in df_tmp.iterrows():
        # Remove system lines from the question
        question_content = row["question"]
        lines = question_content.split("\n")
        user_lines = [l for l in lines if not l.startswith("System:")]
        question_without_system = "\n".join(user_lines)

        # The first-turn answer
        A1 = row["A1"]

        # Build second turn prompt
        new_prompt = f"{question_without_system}\nAssistant: {A1}{ADDED_INSTRUCTION}"
        second_prompts.append(new_prompt)

    # Generate outputs for turn 2
    outputs_turn2 = llm.generate(second_prompts, sampling_params)

    # Merge the second turn results into the same partial_data records
    final_data = []
    for idx, row in enumerate(partial_data):
        A2 = outputs_turn2[idx].outputs[0].text
        gt = row["ground_truth"]
        second_prompt = second_prompts[idx]

        score_corr_A2 = correct_and_format(second_prompt, A2, gt)
        score_format_A2 = format_reward_func(A2, gt)
        score_box_A2 = boxed_format_reward_func(A2)
        total_reward_A2 = score_corr_A2 + score_format_A2 + score_box_A2
        token_length_A2 = len(llm.get_tokenizer().encode(A2))

        merged = {
            **row,  # all A1 stuff
            "A2": A2,
            "A2_correctness": score_corr_A2,
            "A2_token_format": score_format_A2,
            "A2_boxed_format": score_box_A2,
            "A2_total_reward": total_reward_A2,
            "A2_token_length": token_length_A2,
        }
        final_data.append(merged)

    df_final = pd.DataFrame(final_data)

    # Print stats for A2
    print_turn_stats(df_final, turn_label="A2")

    # Confusion matrix
    print_confusion_matrix(df_final)

    # Save everything into a single CSV
    df_final.to_csv(csv_file_path, index=False)
    print(f"\nAll done. Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()
