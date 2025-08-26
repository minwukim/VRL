# run_aime25_vllm.py

import os
import csv
import json
import logging
from typing import Optional, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import SamplingParams

from math_verify import verify, parse  # make sure this module is on PYTHONPATH
from interrupted_inference_vllm_2 import InterruptedInferenceVLLM

# --------------------
# Config
# --------------------
CSV_PATH_INTERRUPTED = "aime25_results_vllm_interrupted_qwen3_8B.csv"
CSV_PATH_NORMAL = "aime25_results_vllm_normal_qwen_8B.csv"

# CSV_PATH_INTERRUPTED = "aime25_results_vllm_interrupted_qwq_32B.csv"
# CSV_PATH_NORMAL = "aime25_results_vllm_normal_qwq_32B.csv"

# Optional: lossless JSONL logs (Fix C)
JSONL_PATH_INTERRUPTED = "aime25_results_vllm_interrupted_qwen3_8B.jsonl"
JSONL_PATH_NORMAL = "aime25_results_vllm_normal_qwen3_8B.jsonl"

MODEL_PATH = "Qwen/Qwen3-8B"      # swap as needed
# MODEL_PATH = "Qwen/QwQ-32B"      # swap as needed

TOP_K_ENTROPY = 20

TOTAL_MAX_NEW_TOKENS = 16000
STEP_MAX_TOKENS = 512

# Qwen-style thinking-mode defaults
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

# vLLM sizing
TENSOR_PARALLEL_SIZE = 2        # set to your number of GPUs
# TENSOR_PARALLEL_SIZE = 4        # set to your number of GPUs

MAX_MODEL_LEN = 16000
GPU_MEM_UTIL = 0.90

# multi-sample config
NUM_SAMPLES = 10
BASE_SEED = 20250825  # any integer; varied per question & sample

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

def prepare_prompt_normal(question: str) -> str:
    # Same as prepare_prompt but WITHOUT the confidence-scores sentence
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "Please reason step by step, and put your final answer within \\boxed{}. <|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


# --------------------
# Fix A: sanitize & force strings before storing
# --------------------
def _sanitize_text(s: Optional[str]) -> str:
    # Coerce to string, remove NULs, normalize newlines; never return None
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def ensure_csv_header_interrupted(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f, escapechar='\\', quoting=csv.QUOTE_ALL)  # Fix B: consistent quoting
            writer.writerow([
                "question_index",
                "sample_index",
                "id",
                "question",
                "ground_truth",
                "prompt",
                "generated_text",
                "token_count",
                "reward",
                "manual_conf_tags",
                "auto_conf_tags",
                "seed",
            ])


def ensure_csv_header_normal(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f, escapechar='\\', quoting=csv.QUOTE_ALL)  # Fix B
            writer.writerow([
                "question_index",
                "sample_index",
                "id",
                "question",
                "ground_truth",
                "prompt",
                "generated_text",
                "token_count",
                "reward",
                "seed",
            ])


def append_row(path: str, row: dict):
    with open(path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, escapechar='\\', quoting=csv.QUOTE_ALL)  # Fix B
        writer.writerow([row[k] for k in row.keys()])


# Fix C: parallel JSONL for lossless storage (optional but recommended)
def append_jsonl(path: str, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def overwrite_with_dataframe(path: str, all_rows: List[dict]):
    df = pd.DataFrame(all_rows)
    # Fix D: ensure generated_text remains a (non-NA) string before write
    if "generated_text" in df.columns:
        df["generated_text"] = (
            df["generated_text"]
            .astype("string")
            .fillna("")
        )
    df.to_csv(
        path,
        index=False,
        escapechar='\\',
        quoting=csv.QUOTE_ALL,  # Fix B
        na_rep=""
    )


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

    # 2) Build interrupted engine + tokenizer
    logging.info(f"Loading vLLM model (single engine for both modes): {MODEL_PATH}")
    engine = InterruptedInferenceVLLM(
        model_name=MODEL_PATH,
        top_k_entropy=TOP_K_ENTROPY,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_mem_util=GPU_MEM_UTIL,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 3) CSVs
    ensure_csv_header_interrupted(CSV_PATH_INTERRUPTED)
    ensure_csv_header_normal(CSV_PATH_NORMAL)
    all_rows_interrupted: List[dict] = []
    all_rows_normal: List[dict] = []

    # 4) Sequential loop with tqdm
    logging.info("Starting sequential generation over AIME25 with vLLM...")
    try:
        for q_idx, (qid, qtext, gt) in enumerate(
            tqdm(zip(ids, questions, ground_truths), total=len(questions), desc="AIME25 vLLM Inference")
        ):
            prompt_interrupted = prepare_prompt(qtext)
            prompt_normal = prepare_prompt_normal(qtext)

            # -----------------------------
            # A) INTERRUPTED multi-sample (NUM_SAMPLES runs, each with distinct seed)
            # -----------------------------
            for s_idx in range(NUM_SAMPLES):
                run_seed = BASE_SEED + (q_idx * 1000) + s_idx
                try:
                    gen_text = engine.generate_with_confidence(
                        prompt=prompt_interrupted,
                        total_max_new_tokens=TOTAL_MAX_NEW_TOKENS,
                        step_max_tokens=STEP_MAX_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        seed=run_seed,  # now supported
                    )
                except Exception as e:
                    logging.error(f"[q={q_idx} s={s_idx}] Interrupted generation error: {e}")
                    gen_text = ""

                # Fix A: sanitize text before any storage
                gen_text = _sanitize_text(gen_text)

                pred_boxed = last_boxed_only_string(gen_text)
                gt_boxed = f"\\boxed{{{gt}}}" if gt is not None else None
                reward = reward_without_format(pred_boxed, gt_boxed)

                token_count = len(tokenizer.encode(gen_text, add_special_tokens=False))

                row_i = {
                    "question_index": q_idx,
                    "sample_index": s_idx,
                    "id": qid,
                    "question": qtext,
                    "ground_truth": gt,
                    "prompt": prompt_interrupted,
                    "generated_text": gen_text,
                    "token_count": token_count,
                    "reward": reward,
                    "manual_conf_tags": engine.manual_tags_added,
                    "auto_conf_tags": engine.self_generated_tags,
                    "seed": run_seed,
                }
                # Fix D: make sure field is a string in the in-memory list as well
                row_i["generated_text"] = _sanitize_text(row_i["generated_text"])

                all_rows_interrupted.append(row_i)
                append_row(CSV_PATH_INTERRUPTED, row_i)
                append_jsonl(JSONL_PATH_INTERRUPTED, row_i)  # Fix C

            if (q_idx + 1) % LOG_EVERY == 0:
                overwrite_with_dataframe(CSV_PATH_INTERRUPTED, all_rows_interrupted)
                logging.info(f"[Interrupted] Saved progress: {q_idx + 1}/{len(questions)} → {CSV_PATH_INTERRUPTED}")

            # -----------------------------
            # B) NORMAL vLLM generation (no interventions), n = NUM_SAMPLES
            #    reuse the SAME engine to avoid allocating a second KV cache
            # -----------------------------
            normal_seed = BASE_SEED + 10_000_000 + q_idx
            sp = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=TOTAL_MAX_NEW_TOKENS,
                n=NUM_SAMPLES,
                stop=["</think>"],  # keep comparable for thinking models
                seed=normal_seed,
                logprobs=0,        # faster baseline
            )
            try:
                outs = engine.llm.generate([prompt_normal], sp)[0]
                for s_idx, seq in enumerate(outs.outputs):
                    text = seq.text or ""
                    # Fix A: sanitize before storage
                    text = _sanitize_text(text)

                    pred_boxed = last_boxed_only_string(text)
                    gt_boxed = f"\\boxed{{{gt}}}" if gt is not None else None
                    reward = reward_without_format(pred_boxed, gt_boxed)
                    token_count = len(tokenizer.encode(text, add_special_tokens=False))

                    row_n = {
                        "question_index": q_idx,
                        "sample_index": s_idx,
                        "id": qid,
                        "question": qtext,
                        "ground_truth": gt,
                        "prompt": prompt_normal,
                        "generated_text": text,
                        "token_count": token_count,
                        "reward": reward,
                        "seed": normal_seed,  # same seed per n-sample batch (vLLM handles branching)
                    }
                    row_n["generated_text"] = _sanitize_text(row_n["generated_text"])  # Fix D

                    all_rows_normal.append(row_n)
                    append_row(CSV_PATH_NORMAL, row_n)
                    append_jsonl(JSONL_PATH_NORMAL, row_n)  # Fix C

                if (q_idx + 1) % LOG_EVERY == 0:
                    overwrite_with_dataframe(CSV_PATH_NORMAL, all_rows_normal)
                    logging.info(f"[Normal] Saved progress: {q_idx + 1}/{len(questions)} → {CSV_PATH_NORMAL}")

            except Exception as e:
                logging.error(f"[q={q_idx}] Normal generation error: {e}")
                overwrite_with_dataframe(CSV_PATH_NORMAL, all_rows_normal)

    finally:
        # Cleanly shut down vLLM workers to reduce warnings
        try:
            engine.llm.shutdown()
        except Exception:
            pass

    # Final flush
    overwrite_with_dataframe(CSV_PATH_INTERRUPTED, all_rows_interrupted)
    overwrite_with_dataframe(CSV_PATH_NORMAL, all_rows_normal)
    logging.info(f"Done.\n  Interrupted rows: {len(all_rows_interrupted)} -> {CSV_PATH_INTERRUPTED}\n  Normal rows: {len(all_rows_normal)} -> {CSV_PATH_NORMAL}")


if __name__ == "__main__":
    main()
