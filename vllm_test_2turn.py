import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Single-file script that performs two-stage evaluation.
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
            {"role": "user", "content": "\nQuestion: " + example["problem"]},
        ],
    }

def build_prompt_from_conversation(conversation):
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation["prompt"])

def extract_ground_truth(text: str) -> str | None:
    return last_boxed_only_string(text)

##############################################
# Reward Functions
##############################################
def correct_reward_func(completion: str, ground_truth: str) -> float:
    return 1.0 if verify(parse(completion), parse(ground_truth)) else 0.0

def boxed_format_reward_func(completion: str) -> float:
    match = re.search(r"\\boxed\\{(.*?)\\}", completion)
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
llm = LLM(model=path)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=4000
)

##############################################
# Main Two-Stage Process
##############################################
def main():
    print("Loading dataset...")
    data = get_math_test_data()
    results = []

    print("\nRunning FIRST TURN for all examples...")
    conversations = []
    ground_truths = []
    for example in data:
        conv = make_conversation(example)
        gt = extract_ground_truth(example["solution"])
        conversations.append(conv)
        ground_truths.append(gt)

    prompts_turn1 = [build_prompt_from_conversation(c) for c in conversations]
    outputs_turn1 = llm.generate(prompts_turn1, sampling_params)

    partial_data = []
    for idx, out in enumerate(outputs_turn1):
        A1 = out.outputs[0].text
        prompt = prompts_turn1[idx]
        gt = ground_truths[idx]

        score_corr_A1 = correct_reward_func(A1, gt)
        score_box_A1 = boxed_format_reward_func(A1)
        total_reward_A1 = score_corr_A1 + score_box_A1
        token_length_A1 = len(llm.get_tokenizer().encode(A1))

        partial_data.append({
            "question": prompt,
            "ground_truth": gt,
            "A1": A1,
            "A1_correctness": score_corr_A1,
            "A1_boxed_format": score_box_A1,
            "A1_total_reward": total_reward_A1,
            "A1_token_length": token_length_A1,
        })

    df_tmp = pd.DataFrame(partial_data)
    print("\nRunning SECOND TURN for all examples...")
    second_prompts = [f"{row['question']}\nAssistant: {row['A1']}{ADDED_INSTRUCTION}" for row in df_tmp.iterrows()]
    outputs_turn2 = llm.generate(second_prompts, sampling_params)

    final_data = []
    for idx, row in enumerate(partial_data):
        A2 = outputs_turn2[idx].outputs[0].text
        gt = row["ground_truth"]
        second_prompt = second_prompts[idx]

        score_corr_A2 = correct_reward_func(A2, gt)
        score_box_A2 = boxed_format_reward_func(A2)
        total_reward_A2 = score_corr_A2 + score_box_A2
        token_length_A2 = len(llm.get_tokenizer().encode(A2))

        merged = {
            **row,
            "A2": A2,
            "A2_correctness": score_corr_A2,
            "A2_boxed_format": score_box_A2,
            "A2_total_reward": total_reward_A2,
            "A2_token_length": token_length_A2,
        }
        final_data.append(merged)

    df_final = pd.DataFrame(final_data)
    df_final.to_csv(csv_file_path, index=False)
    print(f"\nAll done. Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()


# SYSTEM_PROMPT = (
#     "You are an assitant solving a math problem. The user asks a question, and the Assistant solves it. "
#     "Ensure that the final answer in the solution is formatted within \\boxed{}, "
#     "as this formatting is required for correct extraction."
# )