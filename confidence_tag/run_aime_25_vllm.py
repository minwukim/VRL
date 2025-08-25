import os
import csv
import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from math_verify import verify, parse  # ensure available
from interrupted_inference_vllm import InterruptedInferenceVLLM

# --------------------
# Config
# --------------------
CSV_PATH = "aime25_results_vllm.csv"
MODEL_PATH = "Qwen/Qwen3-8B"   # swap as needed
TOP_K_ENTROPY = 10

TOTAL_MAX_NEW_TOKENS = 4096
STEP_MAX_TOKENS = 512

# Qwen-style thinking-mode defaults
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

# vLLM sizing
TENSOR_PARALLEL_SIZE = 2        # set to your number of GPUs
MAX_MODEL_LEN = 16384
GPU_MEM_UTIL = 0.90

LOG_EVERY = 1  # overwrite CSV every N rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------
# AIME helpers
# --------------------
def last_boxed_only_string(string: str) -> Optional[str]:
    if string is None:
        return None
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    start_brace_idx = string.find("{", idx)
    if start_brace_idx < 0:
        return None
    num_left_braces_open = 0
    i = start_brace_idx
    right_brace_idx = -1
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx == -1:
        return None
    return string[idx: right_brace_idx + 1]


def reward_without_format(pred: Optional[str], gt: Optional[str]) -> int:
    if gt is None or pred is None:
        return 0
    try:
        parsed_pred = parse(pred)
        parsed_gt = parse(gt)
        if parsed_pred is not None and parsed_gt is not None:
            return int(verify(parsed_pred, parsed_gt))
        return 0
    except Exception:
        return 0


def prepare_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "Use your confidence scores as a guide: if a step is low confidence, slow down, explain more, and double-check before moving on. <|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


def ensure_csv_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f, escapechar='\\')
            writer.writerow([
                "question_index",
                "id",
                "question",
                "ground_truth",
                "prompt",
                "generated_text",
                "token_count",
                "reward",
                "manual_conf_tags",
                "auto_conf_tags"
            ])


def append_row(path: str, row: dict):
    with open(path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, escapechar='\\')
        writer.writerow([
            row["question_index"],
            row["id"],
            row["question"],
            row["ground_truth"],
            row["prompt"],
            row["generated_text"],
            row["token_count"],
            row["reward"],
            row["manual_conf_tags"],
            row["auto_conf_tags"]
        ])


def overwrite_with_dataframe(path: str, all_rows: list):
    df = pd.DataFrame(all_rows)
    df.to_csv(path, index=False, escapechar='\\')


def main():
    # 1) Load AIME25
    logging.info("Loading AIME25 (split='test') ...")
    try:
        ds = load_dataset("math-ai/aime25", split="test")
    except Exception as e:
        logging.error(f"Failed to load AIME25: {e}")
        raise SystemExit(1)

    questions = [ex["problem"] for ex in ds]
    ground_truths = [str(ex["answer"]) if ex["answer"] is not None else None for ex in ds]
    ids = [ex.get("id", None) for ex in ds]

    # 2) Build vLLM engine + tokenizer
    logging.info(f"Loading vLLM model: {MODEL_PATH}")
    engine = InterruptedInferenceVLLM(
        model_name=MODEL_PATH,
        top_k_entropy=TOP_K_ENTROPY,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_mem_util=GPU_MEM_UTIL,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 3) CSV
    ensure_csv_header(CSV_PATH)
    all_results = []

    # 4) Sequential loop with tqdm
    logging.info("Starting sequential generation over AIME25 with vLLM...")
    try:
        for idx, (qid, qtext, gt) in enumerate(
            tqdm(zip(ids, questions, ground_truths), total=len(questions), desc="AIME25 vLLM Inference")
        ):
            prompt = prepare_prompt(qtext)

            try:
                gen_text = engine.generate_with_confidence(
                    prompt=prompt,
                    total_max_new_tokens=TOTAL_MAX_NEW_TOKENS,
                    step_max_tokens=STEP_MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                )
            except Exception as e:
                logging.error(f"[{idx}] Generation error: {e}")
                gen_text = ""

            pred_boxed = last_boxed_only_string(gen_text)
            gt_boxed = f"\\boxed{{{gt}}}" if gt is not None else None
            reward = reward_without_format(pred_boxed, gt_boxed)

            token_count = len(tokenizer.encode(gen_text, add_special_tokens=False))

            row = {
                "question_index": idx,
                "id": qid,
                "question": qtext,
                "ground_truth": gt,
                "prompt": prompt,
                "generated_text": gen_text,
                "token_count": token_count,
                "reward": reward,
                "manual_conf_tags": engine.manual_tags_added,
                "auto_conf_tags": engine.self_generated_tags,
            }
            all_results.append(row)

            append_row(CSV_PATH, row)
            if (idx + 1) % LOG_EVERY == 0:
                overwrite_with_dataframe(CSV_PATH, all_results)
                logging.info(f"Saved progress: {idx + 1}/{len(questions)} → {CSV_PATH}")
    finally:
        # Try to cleanly shut down vLLM workers to reduce NCCL warnings
        try:
            engine.llm.shutdown()
        except Exception:
            pass

    overwrite_with_dataframe(CSV_PATH, all_results)
    logging.info(f"Done. Wrote {len(all_results)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
