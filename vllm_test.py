import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

# ---------------------------------------------------
# System Prompt and utility functions
# ---------------------------------------------------
SYSTEM_PROMPT = """
Answer the question below in the specified format. 
First, carefully think through the reasoning process. Then, provide a refined solution explaining the steps of the solution. 
Enclose the reasoning process within <think> </think> tags and the final solution within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> solution here </answer>. 
Ensure the final answer in the solution is formatted within \\boxed{}.
"""

def build_prompt(problem: str) -> str:
    return f"{SYSTEM_PROMPT}\nUser: {problem}\nAssistant:"

def extract_ground_truth(text: str) -> str | None:
    return remove_boxed(last_boxed_only_string(text))

# ---------------------------------------------------
# Reward functions
# ---------------------------------------------------
def correctness_reward_func(prompt: str, completion: str, ground_truth: str) -> float:
    # Using compute_score from your custom_MATH_reward module
    return compute_score(completion, ground_truth)

def token_format_reward_func(completion: str) -> float:
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    match = re.search(pattern, completion, re.DOTALL)
    return 0.1 if match else 0.0

def boxed_format_reward_func(completion: str) -> float:
    match = re.search(r"\\boxed\{(.*?)\}", completion)
    return 0.1 if match else 0.0

# ---------------------------------------------------
# Load the entire MATH-500 test dataset and build prompts
# ---------------------------------------------------
def get_math_test_prompts():
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)['test']
    prompts = []
    ground_truths = []
    for example in data:
        prompt = build_prompt(example['problem'])
        gt = extract_ground_truth(example['solution'])
        prompts.append(prompt)
        ground_truths.append(gt)
    return prompts, ground_truths

# ---------------------------------------------------
# Initialize the LLM with your checkpoint
# ---------------------------------------------------
# Replace "./path/to/your/checkpoint" with your actual checkpoint path or model name.

# path = "Qwen/Qwen2.5-3B-Instruct"
path = "./0308-purerl-qwen3b/checkpoint-1875"

llm = LLM(
    model=path
)

# Define sampling parameters (used by LLM.generate() even though no sampling randomness is applied)
sampling_params = SamplingParams(
    temperature=0.0,  # Set temperature to 0 for deterministic output if desired
    top_p=1.0,
    max_tokens=4000
)

# ---------------------------------------------------
# Main Inference and CSV Export
# ---------------------------------------------------
def main():
    # Get prompts and ground truths for the entire test set
    prompts, ground_truths = get_math_test_prompts()
    
    # Generate responses from the LLM for all prompts
    outputs = llm.generate(prompts, sampling_params)
    
    # List to collect results
    results = []
    
    # Process each output
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        # Assuming the first generation is the desired response
        generated_text = output.outputs[0].text
        gt = ground_truths[idx]
        
        # Compute rewards
        score_corr = correctness_reward_func(prompt, generated_text, gt)
        score_format = token_format_reward_func(generated_text)
        score_box = boxed_format_reward_func(generated_text)
        total_reward = score_corr + score_format + score_box
        token_length = len(llm.get_tokenizer().encode(generated_text))

        
        # Append the result to the list
        results.append({
            "question": prompt,
            "ground_truth": gt,
            "response": generated_text,
            "correctness": score_corr,
            "token_format": score_format,
            "boxed_format": score_box,
            "total_reward": total_reward,
            "token_length": token_length
        })
    
    # Create a pandas DataFrame from the results
    df = pd.DataFrame(results)
    
    # Export the DataFrame to a CSV file
    df.to_csv("cp3400_results.csv", index=False)
    
    # Print a summary
    print(f"Processed {len(results)} examples. Results saved to 'vllm_inference_results.csv'.")

if __name__ == "__main__":
    main()
