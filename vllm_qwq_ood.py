import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
# from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse
from pathlib import Path


# ——————————————
# Config
# ——————————————
model_path = "Qwen/QwQ-32B"
csv_train_path = "QwQ_train_365_16_first.csv"

# model_path = "./outputs/qwen2.5-3b-sft-pro/checkpoint-1092"
# csv_train_path = "ood_all_4_second_64.csv"
# csv_train_path = "ood_test_KK_128.csv"
# csv_train_path = "1to64_kk_response.csv"
# csv_train_path = "np128p256_kk.csv"
# csv_train_path = "174_incorrect_response_second.csv"
# csv_train_path = "4all_last246.csv"

column_name = 'kk_not_solved'

# easy_not_solved
# medium_not_solved
# hard_not_solved
# ext_hard_not_solved
# kk_not_solved
# incorrect_not_solved


# csv_test_path = "QwQ_test.csv"
seed = 834902
num_trials = 16
batch_size = 150000
temperature = 0.9
top_p = 1
top_k = 40
min_p = 0.0
presence_penalty = 1.0
tensor_parallel_size = 4

# Prompt template with standardized instruction
# SYSTEM_PROMPT = (
#     "<|im_start|>system\n"
#     "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
#     "<|im_start|>user\n"
#     "{prompt}<|im_end|>\n"
#     "<|im_start|>assistant\n"
# )
SYSTEM_PROMPT = (
    "{prompt} [SEP] "
)


# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The User asks a question, and the Assistant solves it."
#     "The Assistant  first thinks about the reasoning process in the mind and then provides the User with the answer."
#     "The reasoning process is enclosed within <think> </think> and answer is enclosed with in <answer> </answer> tages, respectively,"
#     " i.e., <think> reasoning process here </think> <answer> answer here </answer>./n"
#     "User: {prompt}/nAssitant: <think>"
# )

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

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

def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

def run_evaluation(csv_path, problems, ground_truths, question_indices, dataset_name):
    total_questions = len(problems)
    print(f"\n>>> Starting evaluations on {dataset_name} — {total_questions} questions x {num_trials} trials")

    first_batch = True
    llm = LLM(model=model_path, max_model_len=12000, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()


    for i in range(0, total_questions, batch_size):
        batch_indices = range(i, min(i + batch_size, total_questions))
        batch_prompts = [problems[j] for j in batch_indices]
        batch_ground_truths = [ground_truths[j] for j in batch_indices]
        batch_question_indices = [question_indices[j] for j in batch_indices]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            max_tokens=12000,
            n=num_trials,
            seed=seed,
        )

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"[ERROR] Batch {i}-{i+batch_size} failed: {e}")
            continue

        rows = []
        for j, out in enumerate(outputs):  # each question
            prompt = batch_prompts[j]
            ground_truth = batch_ground_truths[j]
            q_idx = batch_question_indices[j]
            for trial_idx, o in enumerate(out.outputs):  # each of the 256 samples
                response = o.text
                reward = reward_without_format(response, ground_truth)
                rows.append({
                    "trial_index": trial_idx,
                    "question_index": q_idx,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "response": response,
                    "response_length": len(response),
                    "reward": reward
                })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, mode='a', header=first_batch, index=False)
        first_batch = False
        print(f"✓ Saved batch {i}-{i+len(batch_prompts)-1} ({len(rows)} rows)")

    print(f"\n✅ Completed all {num_trials} trials for {dataset_name}")

# ——————————————
# Train set: load from CSV instead of HuggingFace dataset
# ——————————————
# ds_train = pd.read_csv("base_ood_test_questions.csv")
# ds_train = ds_train[ds_train['base_ood'] == 1]
# ds_train = pd.read_csv("base_model_nopass128_pass256_76.csv")
# ds_train = pd.read_csv("base_model_test_question_solution_hit.csv")
ds_train = pd.read_csv("base_model_train_question_solution_hit.csv")
ds_train = ds_train[ds_train['hit'] == 0]

# ds_train = ds_train[ds_train[column_name] == 1]

train_problems = [SYSTEM_PROMPT.format(prompt=p) for p in ds_train['prompt']]
train_truths = [last_boxed_only_string(gt) for gt in ds_train['ground_truth']]
train_question_indices = ds_train['question_index'].tolist()

run_evaluation(csv_train_path, train_problems, train_truths, train_question_indices, dataset_name="OOD_BASE")
