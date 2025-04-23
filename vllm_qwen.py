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
csv_train_path = "QwQ_train.csv"
csv_test_path = "QwQ_test.csv"
seed = 11
num_trials = 32
batch_size = 100000
temperature = 0.9
top_p = 1.0

SYSTEM_PROMPT = "{prompt}"

def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

def run_evaluation(csv_path, problems, ground_truths, dataset_name):
    total_questions = len(problems)
    print(f"\n>>> Starting evaluations on {dataset_name} — {total_questions} questions x {num_trials} trials")

    first_batch = True
    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")

        llm = LLM(model=model_path)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=8000,
            n=1,
            seed=seed + trial,
        )

        for i in range(0, total_questions, batch_size):
            batch_indices = range(i, min(i + batch_size, total_questions))
            batch_prompts = [problems[j] for j in batch_indices]
            batch_ground_truths = [ground_truths[j] for j in batch_indices]
            trial_indices = [trial] * len(batch_prompts)

            try:
                outputs = llm.generate(batch_prompts, sampling_params)
            except Exception as e:
                print(f"[ERROR] Batch {i}-{i+batch_size} failed: {e}")
                continue

            responses = [out.outputs[0].text for out in outputs]
            rewards = [reward_without_format(r, gt) for r, gt in zip(responses, batch_ground_truths)]
            response_lengths = [len(r) for r in responses]

            df = pd.DataFrame({
                "trial_index": trial_indices,
                "question_index": list(batch_indices),
                "prompt": batch_prompts,
                "ground_truth": batch_ground_truths,
                "response": responses,
                "response_length": response_lengths,
                "reward": rewards
            })

            df.to_csv(csv_path, mode='a', header=first_batch, index=False)
            first_batch = False
            print(f"✓ Saved batch {i}-{i+len(batch_prompts)-1} (trial {trial})")

    print(f"\n✅ Completed all {num_trials} trials for {dataset_name}")

# ——————————————
# Train set: apply last_boxed_only_string
# ——————————————
ds_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
train_problems = [SYSTEM_PROMPT.format(prompt=e["problem"]) for e in ds_train]
train_truths = [last_boxed_only_string(e["solution"]) for e in ds_train]

run_evaluation(csv_train_path, train_problems, train_truths, dataset_name="Train Set")

# ——————————————
# Test set: no formatting applied to solution
# ——————————————
ds_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
test_problems = [SYSTEM_PROMPT.format(prompt=e["problem"]) for e in ds_test]
test_truths = [e["solution"] for e in ds_test]

run_evaluation(csv_test_path, test_problems, test_truths, dataset_name="Test Set")
