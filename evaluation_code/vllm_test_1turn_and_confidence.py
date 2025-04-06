import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Settings and Model Initialization
##############################################
model_dir_path = ""
checkpoint_num = "300"  # or update to your desired checkpoint number
model_path = model_dir_path+"/checkpoint-"+checkpoint_num
csv_file_path = "./evaluation_results"+model_dir_path+"checkpoint-"+checkpoint_num+".csv"



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

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
test_dataset = test_dataset.map(lambda x: {
    "prompt": [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x["problem"]}
    ],
    "answer": x["answer"],
    "level": x["level"]
})


def reward_func(completions,answer, **kwargs):
    def check_format_and_correctess(completion, ground_truth):
        response = last_boxed_only_string(completion)
        if response is None:
            return -1
        if verify(parse(response), parse(ground_truth)):   
            return -0.5
        return 1
    return [check_format_and_correctess(c, gt) for c, gt in zip(completions, answer)]


llm = LLM(model=model_path)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1500
)



















# model_path = "./outputs/qwen2.5-3b-grpo-full/checkpoint-400"  # or update to your desired model path
# model_path = "./qwen2.5-3b-grpo-switch/checkpoint-350"  # or update to your desired model path
model_path = "./qwen2.5-3b-grpo-switch-type1-reward/checkpoint-300"
# model_path = "Qwen/Qwen2.5-3B"
# model_path = "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero"
csv_file_path = "./qwen2.5-grpo-switch-csvs/qwen2.5-3b-grpo-switch-type1-reward-checkpoint-350_2stage.csv"


# First turn prompt template
SYSTEM_PROMPT_FIRST = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

# Instruction for the second turn (correction/verification)
ADDED_INSTRUCTION = (
    "A conversation between User and Assistant. Given a question and a corresponding response provided below, the Assistant systematically reviews and explains each step of the reasoning process to verify the correctness of the response."
    "If errors are found, the Assistant identifies and corrects them, then re-solves the problem. If the response is correct, the Assistant confirms it and returns the same final answer."
    "The assistant first thinks about the reasoning process in mind, including verification, correction, and resolving the problem if necessary. Then provides the user with the answer."
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>." 
    "The reasoning process, including verification and correction, is enclosed within <think> </think> tags, while the final solution is enclosed within <answer> </answer> tags. The final answer is formatted within \\boxed{{}} to enable direct extraction for grading."
    "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag."
)

# Second turn prompt template
SYSTEM_PROMPT_SECOND = (
    "{instruction}"
    + "\n\nQuestion:\n{question}"
    + "\n\nResponse:\n<think>{first_answer}"
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
    temperature=0.0,
    top_p=1.0,
    max_tokens=4000
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
    for item in partial_data:
        prompt_second = build_prompt_second(item["problem"], item["A1"])
        prompts_second.append(prompt_second)

    print("\nRunning SECOND TURN for all examples...")
    outputs_turn2 = llm.generate(prompts_second, sampling_params)

    final_data = []
    for idx, item in enumerate(partial_data):
        A2 = outputs_turn2[idx].outputs[0].text
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

    df_final.to_csv(csv_file_path, index=False)
    print(f"\nAll done. Results saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main()



# ##############################################
# # Single-file script that performs two-stage evaluation.
# # 1) First turn: generates an answer for each item.
# # 2) Second turn: re-evaluates each item using the first turn's output.
# # 3) Exports a single CSV file containing the results for both turns.
# # 4) Prints out the same statistics as in the previous code.
# ##############################################

# # Choose your checkpoint path or model name
# path = "Qwen/Qwen2.5-3B-Instruct"

# # The path name for the final CSV file to save the combined results.
# csv_file_path = "two_stage_results.csv"

# ##############################################
# # System Prompt and Additional Instruction
# ##############################################
# # SYSTEM_PROMPT = (
# #     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
# #     "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
# #     "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
# #     "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
# #     "Ensure that the final answer in the solution is formatted within \\boxed{}, "
# #     "as this formatting is required for correct extraction."
# # )

# SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag."

# ADDED_INSTRUCTION = (
#     "\n Given the question and the response provided, go through the reasoning process of the response and check if the response is correct or not. "
#     "Then, try to resolve the problem if incorrect, and return the same final answer if you think it is correct. "
#     "Enclose your reasoning of checking and resolving process within <think> </think> tags and the final solution within <answer> </answer> tags, "
#     "i.e., <think> reasoning process here </think> <answer> solution here </answer>. "
#     "Ensure that the final answer in the solution is formatted within \\boxed{}, as this formatting is required for correct extraction."
# )

# ##############################################
# # Helper Functions
# ##############################################
# def make_conversation(example):
#     return {
#         "prompt": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": "\nQuestion: " + example["problem"]},
#         ],
#     }

# def make_second_turn_conversation(question_text, first_answer):
#     # For the second turn, the system prompt is ADDED_INSTRUCTION,
#     # and the user content is: "Question: <question_text>, Response: <first_answer>"
#     return {
#         "prompt": [
#             {"role": "system", "content": ADDED_INSTRUCTION},
#             {"role": "user", "content": f"\nQuestion: {question_text}\nResponse: {first_answer}"},
#         ],
#     }

# def build_prompt_from_conversation(conversation):
#     # Convert a conversation object to a single prompt string
#     return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation["prompt"])

# def extract_ground_truth(text: str) -> str | None:
#     # Extract the last boxed answer from the reference solution.
#     return last_boxed_only_string(text)

# ##############################################
# # Reward Functions
# ##############################################
# def correct_and_format(prompt: str, completion: str, ground_truth: str) -> float:
#     # Extract the answer enclosed within <answer>...</answer>
#     pattern = r"</think>\n?<answer>([\s\S]*)</answer>"
#     match = re.search(pattern, completion)
#     content = match.group(1) if match else ""
#     return 1.0 if verify(parse(content), parse(ground_truth)) else 0.0

# def correct_reward_func(completion: str, ground_truth: str) -> float:
#     # Checks if the entire completion matches the ground truth
#     return 1.0 if verify(parse(completion), parse(ground_truth)) else 0.0

# def boxed_format_reward_func(completion: str) -> float:
#     # A small reward if there's at least one '\\boxed{}' in the text
#     match = re.search(r"\\boxed\{(.*?)\}", completion)
#     return 0.1 if match else 0.0

# ##############################################
# # Load the MATH-500 Dataset
# ##############################################
# def get_math_test_data():
#     source_name = "HuggingFaceH4/MATH-500"
#     data = load_dataset(source_name, trust_remote_code=True)["test"]
#     return data

# ##############################################
# # LLM Initialization
# ##############################################
# llm = LLM(
#     model=path
# )

# # Deterministic generation
# sampling_params = SamplingParams(
#     temperature=0.0,
#     top_p=1.0,
#     max_tokens=4000
# )

# ##############################################
# # Statistics / Confusion Matrix Helpers
# ##############################################
# def print_turn_stats(df, turn_label="A1"):
#     print(f"\nAverage Metrics for {turn_label}:")
#     avg_corr = df[f"{turn_label}_correctness"].mean()
#     avg_boxed = df[f"{turn_label}_boxed_format"].mean()
#     avg_reward = df[f"{turn_label}_total_reward"].mean()
#     avg_length = df[f"{turn_label}_token_length"].mean()

#     print(f"  Correctness: {avg_corr}")
#     print(f"  Boxed Format: {avg_boxed}")
#     print(f"  Total Reward: {avg_reward}")
#     print(f"  Token Length: {avg_length}")

#     total_examples = len(df)
#     correct_count = (df[f"{turn_label}_correctness"] == 1.0).sum()
#     incorrect_count = (df[f"{turn_label}_correctness"] == 0.0).sum()

#     print(f"\n{turn_label} Accuracy:")
#     print(f"  Correct: {correct_count} ({(correct_count / total_examples)*100:.2f}%)")
#     print(f"  Incorrect: {incorrect_count} ({(incorrect_count / total_examples)*100:.2f}%)")

# def print_confusion_matrix(df):
#     correct_to_correct = ((df["A1_correctness"] == 1.0) & (df["A2_correctness"] == 1.0)).sum()
#     correct_to_incorrect = ((df["A1_correctness"] == 1.0) & (df["A2_correctness"] == 0.0)).sum()
#     incorrect_to_correct = ((df["A1_correctness"] == 0.0) & (df["A2_correctness"] == 1.0)).sum()
#     incorrect_to_incorrect = ((df["A1_correctness"] == 0.0) & (df["A2_correctness"] == 0.0)).sum()

#     print("\nConfusion Matrix (A1 -> A2):")
#     print(f"  Correct to Correct: {correct_to_correct}")
#     print(f"  Correct to Incorrect: {correct_to_incorrect}")
#     print(f"  Incorrect to Correct: {incorrect_to_correct}")
#     print(f"  Incorrect to Incorrect: {incorrect_to_incorrect}")

# ##############################################
# # Main Two-Stage Process
# ##############################################
# def main():
#     print("Loading dataset...")
#     data = get_math_test_data()

#     # Prepare empty list to store final results
#     results = []

#     print("\nRunning FIRST TURN for all examples...")
#     # Build all conversations and gather ground truths
#     conversations = []
#     ground_truths = []
#     for example in data:
#         conv = make_conversation(example)
#         gt = extract_ground_truth(example["solution"])
#         conversations.append(conv)
#         ground_truths.append(gt)

#     # Build prompts for turn 1
#     prompts_turn1 = [build_prompt_from_conversation(c) for c in conversations]

#     # Generate outputs for turn 1
#     outputs_turn1 = llm.generate(prompts_turn1, sampling_params)

#     # Create a partial results list that we can fill in turn2
#     partial_data = []
#     for idx, out in enumerate(outputs_turn1):
#         # Turn 1 answer
#         A1 = out.outputs[0].text
#         prompt = prompts_turn1[idx]
#         gt = ground_truths[idx]

#         score_corr_A1 = correct_reward_func(A1, gt)
#         score_box_A1 = boxed_format_reward_func(A1)
#         total_reward_A1 = score_corr_A1 + score_box_A1
#         token_length_A1 = len(llm.get_tokenizer().encode(A1))

#         partial_data.append({
#             "first_question": prompt,
#             "ground_truth": gt,
#             "A1": A1,
#             "A1_correctness": score_corr_A1,
#             "A1_boxed_format": score_box_A1,
#             "A1_total_reward": total_reward_A1,
#             "A1_token_length": token_length_A1,
#         })

#     # Convert partial_data to a DataFrame so we can do stats
#     df_tmp = pd.DataFrame(partial_data)

#     # Print stats for A1
#     print_turn_stats(df_tmp, turn_label="A1")

#     print("\nRunning SECOND TURN for all examples...")
#     second_prompts = []

#     # Build second turn prompts using the output of turn1, but in a conversation style.
#     for idx, row in df_tmp.iterrows():
#         # We'll remove system lines from the original question so we can isolate the user question.
#         question_content = row["first_question"]
#         lines = question_content.split("\n")
#         user_lines = [l for l in lines if l.startswith("User:" )]
#         if user_lines:
#             # user_lines[0] might look like "User: <stuff>"
#             # so we can strip off "User: " to isolate the question
#             actual_question = user_lines[0].replace("User: ", "").strip()
#         else:
#             # fallback if something unexpected
#             actual_question = question_content

#         # The first-turn answer
#         A1 = row["A1"]

#         # Create a new conversation for turn 2 using our helper
#         second_conv = make_second_turn_conversation(actual_question, A1)
#         # Build the prompt
#         new_prompt = build_prompt_from_conversation(second_conv)
#         second_prompts.append(new_prompt)

#     # Generate outputs for turn 2
#     outputs_turn2 = llm.generate(second_prompts, sampling_params)

#     # Merge the second turn results into the same partial_data records
#     final_data = []
#     for idx, row in enumerate(partial_data):
#         A2 = outputs_turn2[idx].outputs[0].text
#         gt = row["ground_truth"]
#         second_prompt = second_prompts[idx]

#         score_corr_A2 = correct_reward_func(A2, gt)
#         score_box_A2 = boxed_format_reward_func(A2)
#         total_reward_A2 = score_corr_A2 + score_box_A2
#         token_length_A2 = len(llm.get_tokenizer().encode(A2))

#         merged = {
#             **row,  # all A1 stuff
#             "second_question": second_prompt,
#             "A2": A2,
#             "A2_correctness": score_corr_A2,
#             "A2_boxed_format": score_box_A2,
#             "A2_total_reward": total_reward_A2,
#             "A2_token_length": token_length_A2,
#         }
#         final_data.append(merged)

#     df_final = pd.DataFrame(final_data)

#     # Print stats for A2
#     print_turn_stats(df_final, turn_label="A2")

#     # Confusion matrix
#     print_confusion_matrix(df_final)

#     # Save everything into a single CSV
#     df_final.to_csv(csv_file_path, index=False)
#     print(f"\nAll done. Results saved to '{csv_file_path}'.")

# if __name__ == "__main__":
#     main()