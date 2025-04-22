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
model_path = "Qwen/Qwen2.5-3B"  # Path to the model
csv_path = "demo5.csv"
seed = 5

num_trials = 256
batch_size = 100000  # Adjust based on memory capacity
temperature = 0.9
top_p = 1.0

# ——————————————
# Prompt Template
# ——————————————
SYSTEM_PROMPT = "{prompt}"

# ——————————————
# Reward function
# ——————————————
def reward_without_format(s, gt):
    try:
        return int(verify(parse(s), parse(gt)))
    except:
        return 0

# ——————————————
# Load dataset and expand
# ——————————————
print("Loading MATH dataset...")
test_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
# test_ds = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]
base_problems = [ex["problem"] for ex in test_ds]
base_solutions = [last_boxed_only_string(ex["solution"]) for ex in test_ds]

total_questions = len(base_problems)
total_prompts = total_questions * num_trials
print(f"Loaded {total_questions} questions, running {num_trials} trials → {total_prompts} generations")

# ——————————————
# Model + sampling config
# ——————————————
llm = LLM(model=model_path)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=3000,
    n=1,
    seed=seed,
)

# ——————————————
# Run in batches with periodic saving
# ——————————————
first_batch = True

for trial in range(num_trials):
    print(f"\n=== Trial {trial + 1}/{num_trials} ===")

    for i in range(0, total_questions, batch_size):
        batch_indices = range(i, min(i + batch_size, total_questions))
        batch_prompts = [SYSTEM_PROMPT.format(prompt=base_problems[j]) for j in batch_indices]
        batch_ground_truths = [base_solutions[j] for j in batch_indices]
        trial_indices = [trial] * len(batch_prompts)

        # Generate completions
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"[ERROR] Batch {i}-{i+batch_size} failed: {e}")
            continue

        # Process results
        responses = [out.outputs[0].text for out in outputs]
        rewards = [reward_without_format(r, gt) for r, gt in zip(responses, batch_ground_truths)]
        response_lengths = [len(r) for r in responses]

        # Build DataFrame
        df = pd.DataFrame({
            "trial_index": trial_indices,
            "question_index": list(batch_indices),
            "prompt": batch_prompts,
            "ground_truth": batch_ground_truths,
            "response": responses,
            "response_length": response_lengths,
            "reward": rewards
        })

        # Write to CSV (append mode)
        df.to_csv(csv_path, mode='a', header=first_batch, index=False)
        first_batch = False
        print(f"✓ Saved batch {i}-{i+len(batch_prompts)-1} (trial {trial})")

print("\n All trials completed and saved.")
