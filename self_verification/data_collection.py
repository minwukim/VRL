import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
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


# Set up the model and sampling parameters
model_path = "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero"
llm = LLM(model=model_path)
sampling_params = SamplingParams(
    temperature=0.9,  # You can set temperature=0 for deterministic output if desired
    top_p=1.0,
    max_tokens=4000
)

# Get the first 10 prompts and corresponding ground truth answers
prompts, ground_truths = get_math_test_prompts()

results = []


# For each prompt (question) generate 20 responses
for prompt, ground_truth in zip(prompts, ground_truths):
    # Generate 20 responses for the same prompt using list replication
    responses = llm.generate([prompt] * 2, sampling_params)
    for response in responses:
        score = judge_response(response, ground_truth)
        token_length = len(response.split())
        results.append({
            "question": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "judge_response_score": score,
            "token_length": token_length
        })

# Create a DataFrame and export the results as a CSV file.
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("CSV file 'results.csv' has been exported.")

