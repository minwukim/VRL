import re
from datasets import load_dataset
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from vllm import LLMEngine, SamplingParams
from vllm.engine.request import Request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------
# Setup: Prompt construction and ground truth extraction
# ---------------------------------------------------
SYSTEM_PROMPT = """
Answer the question below in the specified format. 
First, carefully think through the reasoning process. Then, provide a refined solution explaining the steps of the solution. 
Enclose the reasoning process within <think> </think> tags and the final solution within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> solution here </answer>. 
Ensure the final answer in the solution is formatted within \\boxed{}.
"""

checkpoint1 = "./0308-purerl-qwen3b/checkpoint-1875"
checkpoint2 = "./0308-purerl-qwen3b/checkpoint-3400"

def build_prompt(problem: str) -> str:
    return f"{SYSTEM_PROMPT}\nUser: {problem}\nAssistant:"

def extract_ground_truth(text: str) -> str | None:
    return remove_boxed(last_boxed_only_string(text))

def get_math_test_dataset():
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)['test']
    # Map the dataset to include the built prompt and extracted ground truth answer
    data = data.map(lambda x: {
        'prompt': build_prompt(x['problem']),
        'ground_truth': extract_ground_truth(x['solution'])
    })
    return data

# ---------------------------------------------------
# Reward functions (same as defined previously)
# ---------------------------------------------------
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    scores = [compute_score(r, answer) for r in responses]
    return scores

def token_format_reward_func(completions, **kwargs):
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.1 if match else 0.0 for match in matches]

def boxed_format_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(r"\\boxed\{(.*?)\}", r) for r in responses]
    return [0.1 if match else 0.0 for match in matches]

def wrap_completion(text: str):
    """Wrap a raw response string to the expected format for reward functions."""
    return [{"content": text}]

# ---------------------------------------------------
# Sampling parameters for generation
# ---------------------------------------------------
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4000,
)

# ---------------------------------------------------
# Initialize three vLLM engines on different GPUs
# ---------------------------------------------------
engine_base = LLMEngine(
    model="Qwen/Qwen2.5-3B-Instruct",
    tokenizer="Qwen/Qwen2.5-3B-Instruct",
    max_seq_len=4000,
    max_batch_size=4,
    device="cuda:0"  # Load on GPU 0
)

engine_ckpt1 = LLMEngine(
    model=checkpoint1,  # Replace with your actual checkpoint path
    tokenizer=checkpoint1,
    max_seq_len=4000,
    max_batch_size=4,
    device="cuda:1"  # Load on GPU 1
)

engine_ckpt2 = LLMEngine(
    model=checkpoint2,  # Replace with your actual checkpoint path
    tokenizer=checkpoint2,
    max_seq_len=4000,
    max_batch_size=4,
    device="cuda:2"  # Load on GPU 2
)

# ---------------------------------------------------
# Function to process a single question: generate responses and compute rewards.
# ---------------------------------------------------
def process_question(example):
    # Retrieve the prompt (already constructed) and ground truth answer
    prompt = example['prompt']
    ground_truth = example['ground_truth']
    
    # Create a generation request
    request = Request(prompt=prompt, sampling_params=sampling_params)
    
    # Generate responses from each engine
    response_base = engine_base.generate([request])[0].text
    response_ckpt1 = engine_ckpt1.generate([request])[0].text
    response_ckpt2 = engine_ckpt2.generate([request])[0].text
    
    # Compute rewards for each model using the three reward functions
    score_corr_base = correctness_reward_func([{"content": prompt}], [wrap_completion(response_base)], ground_truth)[0]
    score_format_base = token_format_reward_func([wrap_completion(response_base)])[0]
    score_box_base = boxed_format_reward_func([wrap_completion(response_base)])[0]
    total_score_base = score_corr_base + score_format_base + score_box_base

    score_corr_ckpt1 = correctness_reward_func([{"content": prompt}], [wrap_completion(response_ckpt1)], ground_truth)[0]
    score_format_ckpt1 = token_format_reward_func([wrap_completion(response_ckpt1)])[0]
    score_box_ckpt1 = boxed_format_reward_func([wrap_completion(response_ckpt1)])[0]
    total_score_ckpt1 = score_corr_ckpt1 + score_format_ckpt1 + score_box_ckpt1

    score_corr_ckpt2 = correctness_reward_func([{"content": prompt}], [wrap_completion(response_ckpt2)], ground_truth)[0]
    score_format_ckpt2 = token_format_reward_func([wrap_completion(response_ckpt2)])[0]
    score_box_ckpt2 = boxed_format_reward_func([wrap_completion(response_ckpt2)])[0]
    total_score_ckpt2 = score_corr_ckpt2 + score_format_ckpt2 + score_box_ckpt2

    return {
        "problem": prompt,
        "ground_truth": ground_truth,
        "responses": {
            "base": response_base,
            "ckpt1": response_ckpt1,
            "ckpt2": response_ckpt2
        },
        "rewards": {
            "base": {
                "correctness": score_corr_base,
                "token_format": score_format_base,
                "boxed_format": score_box_base,
                "total": total_score_base
            },
            "ckpt1": {
                "correctness": score_corr_ckpt1,
                "token_format": score_format_ckpt1,
                "boxed_format": score_box_ckpt1,
                "total": total_score_ckpt1
            },
            "ckpt2": {
                "correctness": score_corr_ckpt2,
                "token_format": score_format_ckpt2,
                "boxed_format": score_box_ckpt2,
                "total": total_score_ckpt2
            }
        }
    }

# ---------------------------------------------------
# Main loop: Process questions concurrently and summarize rewards.
# ---------------------------------------------------
def main():
    # Load the test dataset
    dataset_test = get_math_test_dataset()
    
    # For demonstration purposes, process a subset (e.g., first 10 questions).
    subset = dataset_test.select(range(10))
    
    results = []
    # Use a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_question, example) for example in subset]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing question: {e}")

    # Aggregate rewards for each model
    summary = {
        "base": {"correctness": 0.0, "token_format": 0.0, "boxed_format": 0.0, "total": 0.0, "count": 0},
        "ckpt1": {"correctness": 0.0, "token_format": 0.0, "boxed_format": 0.0, "total": 0.0, "count": 0},
        "ckpt2": {"correctness": 0.0, "token_format": 0.0, "boxed_format": 0.0, "total": 0.0, "count": 0}
    }
    
    for res in results:
        for model in ["base", "ckpt1", "ckpt2"]:
            summary[model]["correctness"] += res["rewards"][model]["correctness"]
            summary[model]["token_format"] += res["rewards"][model]["token_format"]
            summary[model]["boxed_format"] += res["rewards"][model]["boxed_format"]
            summary[model]["total"] += res["rewards"][model]["total"]
            summary[model]["count"] += 1
    
    # Print summary results for each model
    for model in ["base", "ckpt1", "ckpt2"]:
        count = summary[model]["count"]
        print(f"\nSummary for model '{model}':")
        print(f"  Average Correctness: {summary[model]['correctness'] / count if count else 0:.3f}")
        print(f"  Average Token Format: {summary[model]['token_format'] / count if count else 0:.3f}")
        print(f"  Average Boxed Format: {summary[model]['boxed_format'] / count if count else 0:.3f}")
        print(f"  Average Total Reward: {summary[model]['total'] / count if count else 0:.3f}")
        print("=" * 60)

if __name__ == "__main__":
    main()
