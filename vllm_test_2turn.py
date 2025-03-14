import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse


# Choose your checkpoint path or model name.
path = "Qwen/Qwen2.5-3B-Instruct"
# Choose the path name for the CSV file to save the results.
csv_file_path = "2_turns.csv"

# ---------------------------------------------------
# System Prompt and utility functions
# ---------------------------------------------------
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer> "
    "Ensure that the final answer in the solution is formatted within \\boxed{}, as this formatting is required for correct extraction."
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def build_prompt_from_conversation(conversation):
    """
    Convert a conversation object to a single prompt string.
    Each message is prefixed with its capitalized role.
    """
    return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation["prompt"])

def extract_ground_truth(text: str) -> str | None:
    return last_boxed_only_string(text)

# ---------------------------------------------------
# Reward functions
# ---------------------------------------------------
def correct_and_format(prompt: str, completion: str, ground_truth: str) -> float:
    # Extract the answer enclosed within <answer> </answer> tags
    pattern = r"</think>\n?<answer>([\s\S]*)</answer>"
    match = re.search(pattern, completion)
    content = match.group(1) if match else ""
    return 1.0 if verify(parse(content), parse(ground_truth)) else 0.0

def format_reward_func(completion: str, ground_truth: str) -> float:
    return 1.0 if verify(parse(completion), parse(ground_truth)) else 0.0

def boxed_format_reward_func(completion: str) -> float:
    match = re.search(r"\\boxed\{(.*?)\}", completion)
    return 0.1 if match else 0.0

# ---------------------------------------------------
# Load the entire MATH-500 test dataset and build conversations
# ---------------------------------------------------
def get_math_test_conversations():
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)['test']
    conversations = []
    ground_truths = []
    for example in data:
        conv = make_conversation(example)
        gt = extract_ground_truth(example['solution'])
        conversations.append(conv)
        ground_truths.append(gt)
    return conversations, ground_truths

# ---------------------------------------------------
# Initialize the LLM with your checkpoint
# ---------------------------------------------------

llm = LLM(
    model=path
)

# Define sampling parameters (deterministic output with temperature 0)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=4000
)

# ---------------------------------------------------
# Main Inference, Double Run and CSV Export
# ---------------------------------------------------

def main():
    # Get conversation objects and ground truths for the entire test set
    conversations, ground_truths = get_math_test_conversations()
    results = []
    
    # Loop over each example in the test set
    for idx, conv in enumerate(conversations):
        gt = ground_truths[idx]
        
        # ---- Run 1: Generate answer A1 ----
        # Build prompt string from conversation
        original_prompt = build_prompt_from_conversation(conv)
        run1_outputs = llm.generate([original_prompt], sampling_params)
        A1 = run1_outputs[0].outputs[0].text
        
        # ---- Run 2: Re-evaluate A1 and produce A2 ----
        # Remove the system prompt: use only the user message
        question_without_system = conv["prompt"][1]["content"]
        added_instruction = (
            "\n Given the question and the response provided, go through the reasoning process of the response and check if the response is correct or not. "
            "Then, try to resolve the problem if incorrect, and return the same final answer if you think it is correct. "
            "Enclose your reasoning of checking and resolving process within <think> </think> tags and the final solution within <answer> </answer> tags, "
            "i.e., <think> reasoning process here </think> <answer> solution here </answer>. "
            "Ensure that the final answer in the solution is formatted within \\boxed{}, as this formatting is required for correct extraction."
        )
        run2_prompt = f"User: {question_without_system}\nAssistant: {A1}{added_instruction}"
        run2_outputs = llm.generate([run2_prompt], sampling_params)
        A2 = run2_outputs[0].outputs[0].text

        # ---- Compute reward values and token lengths for A1 and A2 ----
        score_corr_A1 = correct_and_format(original_prompt, A1, gt)
        score_format_A1 = format_reward_func(A1, gt)
        score_box_A1 = boxed_format_reward_func(A1)
        total_reward_A1 = score_corr_A1 + score_format_A1 + score_box_A1
        token_length_A1 = len(llm.get_tokenizer().encode(A1))
        
        score_corr_A2 = correct_and_format(run2_prompt, A2, gt)
        score_format_A2 = format_reward_func(A2, gt)
        score_box_A2 = boxed_format_reward_func(A2)
        total_reward_A2 = score_corr_A2 + score_format_A2 + score_box_A2
        token_length_A2 = len(llm.get_tokenizer().encode(A2))
        
        # ---- Save results for this example ----
        results.append({
            "question": original_prompt,
            "ground_truth": gt,
            "A1": A1,
            "A2": A2,
            "A1_correctness": score_corr_A1,
            "A1_token_format": score_format_A1,
            "A1_boxed_format": score_box_A1,
            "A1_total_reward": total_reward_A1,
            "A1_token_length": token_length_A1,
            "A2_correctness": score_corr_A2,
            "A2_token_format": score_format_A2,
            "A2_boxed_format": score_box_A2,
            "A2_total_reward": total_reward_A2,
            "A2_token_length": token_length_A2,
        })
    
    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results)
    
    # ---- Compute average metrics for A1 and A2 ----
    avg_A1_correctness = df["A1_correctness"].mean()
    avg_A1_token_format = df["A1_token_format"].mean()
    avg_A1_boxed_format = df["A1_boxed_format"].mean()
    avg_A1_total_reward = df["A1_total_reward"].mean()
    avg_A1_token_length = df["A1_token_length"].mean()
    
    avg_A2_correctness = df["A2_correctness"].mean()
    avg_A2_token_format = df["A2_token_format"].mean()
    avg_A2_boxed_format = df["A2_boxed_format"].mean()
    avg_A2_total_reward = df["A2_total_reward"].mean()
    avg_A2_token_length = df["A2_token_length"].mean()
    
    print("Average Metrics for A1:")
    print(f"  Correctness: {avg_A1_correctness}")
    print(f"  Token Format: {avg_A1_token_format}")
    print(f"  Boxed Format: {avg_A1_boxed_format}")
    print(f"  Total Reward: {avg_A1_total_reward}")
    print(f"  Token Length: {avg_A1_token_length}")
    
    print("\nAverage Metrics for A2:")
    print(f"  Correctness: {avg_A2_correctness}")
    print(f"  Token Format: {avg_A2_token_format}")
    print(f"  Boxed Format: {avg_A2_boxed_format}")
    print(f"  Total Reward: {avg_A2_total_reward}")
    print(f"  Token Length: {avg_A2_token_length}")
    
    # ---- Accuracy Statistics ----
    total_examples = len(df)
    A1_correct_count = (df["A1_correctness"] == 1.0).sum()
    A1_incorrect_count = (df["A1_correctness"] == 0.0).sum()
    A2_correct_count = (df["A2_correctness"] == 1.0).sum()
    A2_incorrect_count = (df["A2_correctness"] == 0.0).sum()
    
    print("\nA1 Accuracy:")
    print(f"  Correct: {A1_correct_count} ({(A1_correct_count/total_examples)*100:.2f}%)")
    print(f"  Incorrect: {A1_incorrect_count} ({(A1_incorrect_count/total_examples)*100:.2f}%)")
    
    print("\nA2 Accuracy:")
    print(f"  Correct: {A2_correct_count} ({(A2_correct_count/total_examples)*100:.2f}%)")
    print(f"  Incorrect: {A2_incorrect_count} ({(A2_incorrect_count/total_examples)*100:.2f}%)")
    
    # ---- Confusion Matrix: Transitions from A1 to A2 ----
    correct_to_correct = ((df["A1_correctness"] == 1.0) & (df["A2_correctness"] == 1.0)).sum()
    correct_to_incorrect = ((df["A1_correctness"] == 1.0) & (df["A2_correctness"] == 0.0)).sum()
    incorrect_to_correct = ((df["A1_correctness"] == 0.0) & (df["A2_correctness"] == 1.0)).sum()
    incorrect_to_incorrect = ((df["A1_correctness"] == 0.0) & (df["A2_correctness"] == 0.0)).sum()
    
    print("\nConfusion Matrix (A1 -> A2):")
    print(f"  Correct to Correct: {correct_to_correct}")
    print(f"  Correct to Incorrect: {correct_to_incorrect}")
    print(f"  Incorrect to Correct: {incorrect_to_correct}")
    print(f"  Incorrect to Incorrect: {incorrect_to_incorrect}")
    
    # ---- Save results to CSV ----
    df.to_csv(csv_file_path, index=False)
    print(f"\nProcessed {len(results)} examples. Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()
