import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from math_verify import verify, parse

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def build_prompt(question: str) -> str:
    return f"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

def judge_response(response, ground_truth):
    answer = last_boxed_only_string(response)
    if answer is None:
        return 0.0
    if verify(parse(answer), parse(ground_truth)):
        return 1.0
    return 0.0

def get_math_test_prompts():
    source_name = "DigitalLearningGmbH/MATH-lighteval"
    data = load_dataset(source_name, trust_remote_code=True)['train']
    prompts = []
    ground_truths = []
    # Select only the first 10 examples
    for example in data.select(range(10)):
        prompt = build_prompt(example['problem'])
        gt = last_boxed_only_string(example['solution'])
        prompts.append(prompt)
        ground_truths.append(gt)
    return prompts, ground_truths

# Set up the model with dtype explicitly set to float16 for Tesla V100 GPUs
model_path = "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero"
llm = LLM(model=model_path, dtype=torch.float16)

sampling_params = SamplingParams(
    temperature=0.9,  # Set temperature=0 for deterministic output if desired
    top_p=1.0,
    max_tokens=4000
)

# Retrieve prompts and ground truth answers for the first 10 examples
prompts, ground_truths = get_math_test_prompts()

# Create a single list containing each prompt repeated 20 times
num_samples = 20
all_prompts = []
for prompt in prompts:
    all_prompts.extend([prompt] * num_samples)

# Generate outputs for all prompts in one call
outputs = llm.generate(all_prompts, sampling_params)

# Process the outputs and compile results in a single list
results = []
for idx, output in enumerate(outputs):
    # Determine the corresponding original question index
    question_idx = idx // num_samples
    prompt_text = output.prompt
    # Assuming the first generation is the desired response
    generated_text = output.outputs[0].text
    score = judge_response(generated_text, ground_truths[question_idx])
    token_length = len(generated_text.split())
    results.append({
        "question": prompt_text,
        "response": generated_text,
        "ground_truth": ground_truths[question_idx],
        "judge_response_score": score,
        "token_length": token_length
    })

# Convert results to a DataFrame and export to CSV
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("CSV file 'results.csv' has been exported.")
