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
# model_path = "Qwen/QwQ-32B"
# csv_train_path = "QwQ_train.csv"

# model_path = "./qwq_distill_cps/0428-base-distill-qwq-ext-hard-response/checkpoint-2140"
# model_path = "./qwq_distill_cps/0428-base-distill-qwq-hard-response/checkpoint-2000"
# model_path = "./qwq_distill_cps/checkpoint-2500"
# model_path = "./qwq_distill_cps/4-all-checkpoint/4-all-checkpoint"
# model_path = "./qwq_distill_cps/0428-base-distill-qwq-easy-response/checkpoint-2500"
# model_path = "./qwq_distill_cps/qwq_wrong/checkpoint-2500"
# model_path = "Qwen/Qwen2.5-3B"
# model_path = "0421-qwen3b-question-only-no-format/checkpoint-150"

# model_path = "Qwen/Qwen2.5-Math-1.5B"

model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# model_path = "./outputs/qwen2.5-3b-sft-pro/checkpoint-1092"
# csv_train_path = "ood_all_4_second_64.csv"
# csv_train_path = "ood_test_KK_128.csv"
# csv_train_path = "1to64_kk_response.csv"
# csv_train_path = "np128p256_kk.csv"
# csv_train_path = "174_incorrect_response_second.csv"
# csv_train_path = "4all_last246.csv"
# csv_train_path = "AIME25_256.csv"
# csv_train_path = "AIME25_base_256.csv"
# csv_train_path = "AIME25_incorrect_256.csv"

csv_train_path = "AIME25_256_1.5B_MATH_distilled_second.csv"

# csv_train_path = "AIME25_256_1.5B_MATH.csv"




# easy_not_solved
# medium_not_solved
# hard_not_solved
# ext_hard_not_solved
# kk_not_solved
# incorrect_not_solved


# csv_test_path = "QwQ_test.csv"
seed = 24
num_trials = 128
batch_size = 150000
temperature = 0.6
top_p = 1
top_k = 40
min_p = 0.0
presence_penalty = 1.0
tensor_parallel_size = 2

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

SYSTEM_PROMPT = (
    "{prompt}"
)



SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it."
    "The Assistant  first thinks about the reasoning process in the mind and then provides the User with the answer."
    "The reasoning process is enclosed within <think> </think> and answer is enclosed with in <answer> </answer> tages, respectively,"
    " i.e., <think> reasoning process here </think> <answer> answer here </answer>./n"
    "User: {prompt}/nAssistant: <think>"
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
    llm = LLM(model=model_path, max_model_len=15000, tensor_parallel_size=tensor_parallel_size)
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
            max_tokens=15000,
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
        df.to_csv(csv_path, mode='a', header=first_batch, index=False, quoting=1, escapechar='\\')
        first_batch = False
        print(f"✓ Saved batch {i}-{i+len(batch_prompts)-1} ({len(rows)} rows)")

    print(f"\n✅ Completed all {num_trials} trials for {dataset_name}")

# ——————————————
# Train set: load from CSV instead of HuggingFace dataset
# ——————————————

ds_train= load_dataset("math-ai/aime25", split="test")

train_problems = [SYSTEM_PROMPT.format(prompt=ex['problem']) for ex in ds_train]
train_truths = [ex['answer'] for ex in ds_train]
train_question_indices = [ex['id'] for ex in ds_train]  # or use range(len(dataset)) if 'id' is not numeric

run_evaluation(csv_train_path, train_problems, train_truths, train_question_indices, dataset_name="OOD_BASE")
