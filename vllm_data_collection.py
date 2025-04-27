#!/usr/bin/env python3
# generate_math_train_test.py
"""
Do MATH generation on **one GPU** for both the *train* and *test* sets:

* Dataset 1  (train) :  DigitalLearningGmbH/MATH-lighteval
* Dataset 2  (test)  :  HuggingFaceH4/MATH-500

For each GPU we run 32 trials (⇒ 256 trials total on 8 GPUs):
  – run the 32 trials on the **train** set first  
  – then run the 32 trials on the **test**  set

Each dataset gets its **own** CSV, e.g.

    file1_train.csv   file1_test.csv      (GPU-0)
    file2_train.csv   file2_test.csv      (GPU-1)
    …
    file8_train.csv   file8_test.csv      (GPU-7)

Launch all eight GPUs with the small shell loop at the end of this file.
"""

# ───────────────────────────── imports ─────────────────────────────
import argparse, os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

from obsolete.custom_MATH_reward import last_boxed_only_string
from math_verify import verify, parse

# ───────────────────── configuration constants ─────────────────────
# MODEL_PATH      = "Qwen/Qwen2.5-3B"
MODEL_PATH      = "./0421-qwen3b-question-only-no-format/checkpoint-200"      # 3B model
TRIALS_PER_GPU  = 32                # 32 × 8 = 256
TEMPERATURE     = 0.9
TOP_P           = 1.0
MAX_TOKENS      = 4000
BASE_SEED       = 42                # distinct seed space per GPU later
FILE_PREFIX      = "cp200"      # prefix for CSV filenames
SYSTEM_PROMPT   = "{prompt}"        # no special system prefix for now

# ──────────────────────────── helpers ──────────────────────────────
def reward_without_format(pred: str, truth: str) -> int:
    """Exact-match reward without boxing."""
    try:
        return int(verify(parse(pred), parse(truth)))
    except Exception:
        return 0

def run_trials(
    *, llm: LLM, problems: list[str], truths: list[str],
    csv_path: Path, gpu_id: int, seed_offset: int
):
    """
    Run TRIALS_PER_GPU trials on `problems` and append to `csv_path`.
    seed_offset distinguishes train vs. test seeds.
    """
    first_write = not csv_path.exists()
    n = len(problems)

    for t in range(TRIALS_PER_GPU):
        seed = BASE_SEED + seed_offset + t
        sp = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            n=1,
            seed=seed,
        )
        print(f"[GPU{gpu_id}] {csv_path.stem}: trial {t+1}/{TRIALS_PER_GPU}  (seed={seed})")

        outs = llm.generate(problems, sp)
        responses = [o.outputs[0].text for o in outs]
        rewards   = [reward_without_format(r, gt) for r, gt in zip(responses, truths)]

        df = pd.DataFrame({
            "gpu_id":          gpu_id,
            "trial_index":     t,
            "question_index":  list(range(n)),
            "prompt":          problems,
            "ground_truth":    truths,
            "response":        responses,
            "response_length": [len(r) for r in responses],
            "reward":          rewards,
            "seed":            seed,
        })
        df.to_csv(csv_path, mode="a", header=first_write, index=False, quoting=1, escapechar="\\")

        # df.to_csv(csv_path, mode="a", header=first_write, index=False)
        first_write = False
        print(f"[GPU{gpu_id}]   ✓ wrote {n} rows")


# ──────────────────────────── main ─────────────────────────────────
def main():
    # ---- CLI:  which GPU am I? ------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", type=int, required=True, help="0–7")
    args = ap.parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # ---- load both datasets *once* -------------------------------
    print(f"[GPU{gpu_id}] loading datasets…")

    # train set
    ds_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    train_problems = [SYSTEM_PROMPT.format(prompt=e["problem"])     for e in ds_train]
    train_truths   = [last_boxed_only_string(e["solution"])         for e in ds_train]

    # test set
    ds_test  = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_problems  = [SYSTEM_PROMPT.format(prompt=e["problem"])     for e in ds_test]
    test_truths    = [e["solution"]                                 for e in ds_test]

    # train set
    ds_test_5000 = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    test_5000_problems = [SYSTEM_PROMPT.format(prompt=e["problem"])     for e in ds_test_5000]
    test_5000_truths   = [last_boxed_only_string(e["solution"])         for e in ds_test_5000]

    # ---- model (locked to this GPU) ------------------------------
    llm = LLM(model=MODEL_PATH)

    # ---- run TRAIN first, then TEST ------------------------------
    run_trials(
        llm=llm,
        problems=train_problems,
        truths=train_truths,
        csv_path=Path(f"{FILE_PREFIX}_file{gpu_id+1}_train.csv"),
        gpu_id=gpu_id,
        seed_offset=(gpu_id+1) * 100000             # keep seed spaces disjoint
    )

    run_trials(
        llm=llm,
        problems=test_problems,
        truths=test_truths,
        csv_path=Path(f"{FILE_PREFIX}_file{gpu_id+1}_test.csv"),
        gpu_id=gpu_id,
        seed_offset=100000000 + gpu_id * 10_000 # separate seed range for test set
    )

    run_trials(
        llm=llm,
        problems=test_5000_problems,
        truths=test_5000_truths,
        csv_path=Path(f"{FILE_PREFIX}_file{gpu_id+1}_test5000.csv"),
        gpu_id=gpu_id,
        seed_offset=(gpu_id+1) * 100000             # keep seed spaces disjoint
    )

    print(f"[GPU{gpu_id}] all trials finished.")


if __name__ == "__main__":
    main()


# ─────────────────────────── launcher  ─────────────────────────────
# Save the three lines below as run_all.sh and `chmod +x run_all.sh`
#
#   for g in $(seq 0 7); do
#     CUDA_VISIBLE_DEVICES=$g nohup \
#       python generate_math_train_test.py --gpu-id $g \
#       > log_gpu$g.txt 2>&1 &
#   done
#   wait
#
# Every GPU now runs 32 trials on the *train* set first,
# then 32 trials on the *test* set, writing to its own CSVs.
