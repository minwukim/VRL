import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

# Load trained model
model_path = "Qwen/Qwen2.5-3B-Instruct" # Update with actual model path 
file_path = "3B_vanilla.csv"

llm = LLM(model=model_path)

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
And your final answer will be extracted automatically by the \boxed{{}} tag.
{prompt}
Assistant: <think>
"""

def build_prompt(problem: str) -> str:
    return SYSTEM_PROMPT.format(prompt=problem)

def extract_boxed_answer(solution: str) -> str | None:
    return last_boxed_only_string(solution)

def reward_correct_and_format(completions, answer, **kwargs):
    matches = [re.search(r"</think>\n?<answer>([\s\S]*)</answer>", completion) for completion in completions]
    completions = [match.group(1) if match else "" for match in matches]
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if verify(parse(c), parse(gt)) else 0.0 for c, gt in zip(contents, answer)]

def reward_correct(completions, answer, **kwargs):
    return [1.0 if verify(parse(c), parse(gt)) else 0.0 for c, gt in zip(completions, answer)]

def get_math_test_prompts():
    source_name = "HuggingFaceH4/MATH-500"
    data = load_dataset(source_name, trust_remote_code=True)['test']
    prompts = []
    ground_truths = []
    for example in data:
        prompt = build_prompt(example['problem'])
        gt = extract_boxed_answer(example['solution'])
        prompts.append(prompt)
        ground_truths.append(gt)
    return prompts, ground_truths

def main():
    prompts, ground_truths = get_math_test_prompts()
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4000
    )
    outputs = llm.generate(prompts, sampling_params)
    
    total_correctness = 0.0
    total_correct_and_format = 0.0
    total_length = 0
    num_samples = len(outputs)
    
    results = []
    
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[idx]
        
        score_corr = reward_correct([generated_text], [gt])[0]
        score_correct_and_format = reward_correct_and_format([generated_text], [gt])[0]
        token_length = len(llm.get_tokenizer().encode(generated_text))
        
        total_correctness += score_corr
        total_correct_and_format += score_correct_and_format
        total_length += token_length
        
        results.append({
            "Prompt": prompts[idx],
            "Generated Answer": generated_text,
            "Ground Truth": gt,
            "Correctness_Score": score_corr,
            "Correctness_and_Format_Score": score_correct_and_format,
            "Response Length": token_length
        })
    
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    
    total_correctness = df["Correctness_Score"].sum()
    total_correct_and_format = df["Correctness_and_Format_Score"].sum()
    total_length = df["Response Length"].sum()

    print(f"Average Correctness Score: {total_correctness / num_samples:.2f}")
    print(f"Average Correctness and Format Score: {total_correct_and_format / num_samples:.2f}")
    print(f"Average Response Length: {total_length / num_samples:.2f} tokens")

if __name__ == "__main__":
    main()
