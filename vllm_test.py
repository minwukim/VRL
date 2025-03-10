import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import your custom reward functions
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
def correctness_reward_func(prompt, completion, ground_truth) -> float:
    # Wrap completion in the expected format for reward functions
    wrapped = [{"content": completion}]
    # Using compute_score from your custom_MATH_reward module
    return compute_score(completion, ground_truth)

def token_format_reward_func(completion) -> float:
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    match = re.search(pattern, completion, re.DOTALL)
    return 0.1 if match else 0.0

def boxed_format_reward_func(completion) -> float:
    match = re.search(r"\\boxed\{(.*?)\}", completion)
    return 0.1 if match else 0.0

# ---------------------------------------------------
# Load the MATH-500 test dataset and build prompts
# ---------------------------------------------------
def get_math_test_prompts(num_examples: int = 10):
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)['test']
    # Select a subset for demonstration
    data = data.select(range(num_examples))
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

path = "Qwen/Qwen2.5-3B-Instruct"

# Replace "./path/to/your/checkpoint" with your actual checkpoint path or model name.
llm = LLM(
    model=path,  # Path to your GRPO-trained model
    max_seq_len=4000,
    max_batch_size=4
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1000
)

# ---------------------------------------------------
# Main Inference and Reward Computation Loop
# ---------------------------------------------------
def main():
    # Get prompts and corresponding ground truths from the dataset
    prompts, ground_truths = get_math_test_prompts(num_examples=10)
    
    # Generate responses from the LLM. The LLM.generate() method accepts a list of prompt strings.
    outputs = llm.generate(prompts, sampling_params)
    
    # Initialize cumulative rewards
    total_correctness = 0.0
    total_token_format = 0.0
    total_boxed_format = 0.0
    
    # Process each output
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        # Assuming the first generation is what you want:
        generated_text = output.outputs[0].text
        gt = ground_truths[idx]
        
        # Compute rewards
        score_corr = correctness_reward_func(prompt, generated_text, gt)
        score_format = token_format_reward_func(generated_text)
        score_box = boxed_format_reward_func(generated_text)
        total_reward = score_corr + score_format + score_box
        
        total_correctness += score_corr
        total_token_format += score_format
        total_boxed_format += score_box
        
        # Print the individual results
        print(f"Prompt:\n{prompt}")
        print(f"Ground Truth:\n{gt}")
        print(f"Generated text:\n{generated_text}")
        print(f"Rewards -> Correctness: {score_corr:.3f}, Token Format: {score_format:.3f}, Boxed Format: {score_box:.3f}, Total: {total_reward:.3f}")
        print("-" * 50)
    
    num = len(outputs)
    print("\nSummary of Rewards:")
    print(f"Average Correctness: {total_correctness/num:.3f}")
    print(f"Average Token Format: {total_token_format/num:.3f}")
    print(f"Average Boxed Format: {total_boxed_format/num:.3f}")
    print(f"Average Total Reward: {(total_correctness+total_token_format+total_boxed_format)/num:.3f}")

if __name__ == "__main__":
    main()
