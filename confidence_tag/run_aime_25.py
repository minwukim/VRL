import os
import csv
import logging
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from math_verify import verify, parse  # ensure this is installed/available
from interrupted_inference import InterruptedInference

# --------------------
# Config
# --------------------
CSV_PATH = "aime25_results.csv"
MODEL_PATH = "Qwen/Qwen3-8B"  # swap to any HF model you like
TOP_K_ENTROPY = 20
MAX_NEW_TOKENS = 16000

# Qwen3 "thinking mode" defaults (works fine for other models too)
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0

LOG_EVERY = 1  # overwrite CSV every N items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------
# AIME-style eval helpers
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

    # expected fields: problem, answer, id
    questions = [ex["problem"] for ex in ds]
    ground_truths = [str(ex["answer"]) if ex["answer"] is not None else None for ex in ds]
    ids = [ex.get("id", None) for ex in ds]

    # 2) Initialize engine (sequential; no vLLM)
    logging.info(f"Loading model: {MODEL_PATH}")
    try:
        engine = InterruptedInference(model_name=MODEL_PATH, top_k_entropy=TOP_K_ENTROPY)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error("Ensure model access and login via `huggingface-cli login` if needed.")
        raise SystemExit(1)

    # 3) CSV setup
    ensure_csv_header(CSV_PATH)
    all_results = []

    logging.info("Starting sequential generation over AIME25...")
    for idx, (qid, qtext, gt) in enumerate(
        tqdm(zip(ids, questions, ground_truths), total=len(questions), desc="AIME25 Inference")
    ):
        prompt = prepare_prompt(qtext)

        try:
            gen_text = engine.generate_with_confidence(
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                min_p=MIN_P,
            )
        except Exception as e:
            logging.error(f"[{idx}] Generation error: {e}")
            gen_text = ""

        # Extract pred as last \boxed{...}; wrap GT similarly for verifier
        pred_boxed = last_boxed_only_string(gen_text)
        gt_boxed = f"\\boxed{{{gt}}}" if gt is not None else None
        reward = reward_without_format(pred_boxed, gt_boxed)

        # Proper token count using the tokenizer
        token_count = len(engine.tokenizer.encode(gen_text, add_special_tokens=False))

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

        # Append quickly, also periodically overwrite with full DF (robustness)
        append_row(CSV_PATH, row)
        if (idx + 1) % LOG_EVERY == 0:
            overwrite_with_dataframe(CSV_PATH, all_results)
            logging.info(f"Saved progress: {idx + 1}/{len(questions)} → {CSV_PATH}")

    # Final write
    overwrite_with_dataframe(CSV_PATH, all_results)
    logging.info(f"Done. Wrote {len(all_results)} rows to {CSV_PATH}")


if __name__ == "__main__":
    # torch.manual_seed(0)  # uncomment for deterministic-ish runs
    main()
