import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from obsolete.custom_MATH_reward import remove_boxed, last_boxed_only_string
from math_verify import verify, parse
from pathlib import Path

# ——————————————
# Config
# ——————————————
model_path = "Qwen/QwQ-32B"
# csv_train_path = "QwQ_train.csv"
# model_path = "Qwen/Qwen2.5-32B-Instruct"
csv_train_path = "QwQ_model_4_first16.csv"
# csv_test_path = "QwQ_test.csv"
seed = 678239
num_trials = 16
batch_size = 150000
temperature = 0.6
top_p = 0.95
top_k = 40
min_p = 0.0
presence_penalty = 1.0
tensor_parallel_size =4

# Prompt template with standardized instruction
# SYSTEM_PROMPT = (
#     "{prompt}"
# )

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

def run_evaluation(csv_path, problems, ground_truths, dataset_name):
    total_questions = len(problems)
    print(f"\n>>> Starting evaluations on {dataset_name} — {total_questions} questions x {num_trials} trials")

    first_batch = True
    llm = LLM(model=model_path, max_model_len=10000, tensor_parallel_size=tensor_parallel_size)

    for i in range(0, total_questions, batch_size):
        batch_indices = range(i, min(i + batch_size, total_questions))
        batch_prompts = [problems[j] for j in batch_indices]
        batch_ground_truths = [ground_truths[j] for j in batch_indices]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            max_tokens=10000,
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
            q_idx = batch_indices[j]
            for trial_idx, o in enumerate(out.outputs):  # each of the 16 samples
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
        # df.to_csv(csv_path, mode='a', header=first_batch, index=False)
        df.to_csv(csv_path, mode='a', header=first_batch, index=False, quoting=1, escapechar='\\')
        first_batch = False
        print(f"✓ Saved batch {i}-{i+len(batch_prompts)-1} ({len(rows)} rows)")

    print(f"\n✅ Completed all {num_trials} trials for {dataset_name}")

# ——————————————
# Train set: apply last_boxed_only_string
# ——————————————
# ds_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
# train_problems = [SYSTEM_PROMPT.format(prompt=e["problem"]) for e in ds_train]
# train_truths = [last_boxed_only_string(e["solution"]) for e in ds_train]

ds_train = pd.read_csv("MODEL4_data.csv")
ds_train = ds_train[
    (ds_train['3b_zero_hit'] == 1) |
    (ds_train['1.5b_zero_hit'] == 1) |
    (ds_train['qwq_below_4'] == 1)
]

train_problems = [SYSTEM_PROMPT.format(prompt=e) for e in ds_train['prompt']]
train_truths = [last_boxed_only_string(e) for e in ds_train['ground_truth']]

run_evaluation(csv_train_path, train_problems, train_truths, dataset_name="Train Set")

