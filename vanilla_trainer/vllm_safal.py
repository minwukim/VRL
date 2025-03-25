import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

from eval_datasets import get_kk_test_prompts

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", help="path to model")
parser.add_argument("file", help="output file")
args = parser.parse_args()

# Load trained model
model_path = args.model # Update with actual model path 
file_path = args.file

llm = LLM(model=model_path, max_model_len=5000, gpu_memory_utilization=0.7)

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>
"""

def build_prompt(problem: str) -> str:
    return SYSTEM_PROMPT.format(prompt=problem)

def extract_boxed_answer(solution: str) -> str | None:
    return last_boxed_only_string(solution)

def reward_correct(completions, answer, **kwargs):
    # check if the strings ends with </think><answer>[boxed answer]</answer>
    def check_format(s, gt):
        pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
        if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
            # incorrect amount of tokens
            return -2 
        match = re.search(pattern, s, re.DOTALL)
        # if answer doesn't match provided format
        if not match: return -2

        # answer format is correct now
        # look for boxed tag
        ext_string = last_boxed_only_string(match.group(1))
        if ext_string is None: return -1   #No boxed tag found
        
        # if correct, then reward 2
        if verify(parse(ext_string), parse(gt)): return 2
        else: return -0.5 # extracted but incorrect then reward -0.5

    return [check_format(c, gt) for c, gt in zip(completions, answer)]

#def reward_correct_and_format(completions, answer, **kwargs):
#    matches = [re.search(r"</think>\n?<answer>([\s\S]*)</answer>", completion) for completion in completions]
#    completions = [match.group(1) if match else "" for match in matches]
#    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
#    contents = [match.group(1) if match else "" for match in matches]
#    return [1.0 if verify(parse(c), parse(gt)) else 0.0 for c, gt in zip(contents, answer)]
#
#def reward_correct(completions, answer, **kwargs):
#    return [1.0 if verify(parse(c), parse(gt)) else 0.0 for c, gt in zip(completions, answer)]

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


def get_ob_test_prompts():
    source_name = "Hothan/OlympiadBench"
    #data = load_dataset(source_name, "TP_TO_maths_en_COMP", trust_remote_code=True)['train']
    
    data = load_dataset(source_name, "OE_TO_maths_en_COMP", trust_remote_code=True)['train']
    prompts = []
    ground_truths = []
    for example in data:
        prompt = build_prompt(example['question'])
        gt = example['final_answer'][0]
        #gt = example['solution'][0]
        prompts.append(prompt)
        ground_truths.append(gt)
    return prompts, ground_truths

def get_aime_prompts():
    source_name = "Maxwell-Jia/AIME_2024"
    data = load_dataset(source_name, trust_remote_code=True)['train']
    prompts = []
    ground_truths = []
    for example in data:
        prompt = build_prompt(example['Problem'])
        gt = (example['Answer'])
        prompts.append(prompt)
        ground_truths.append(gt)
    return prompts, ground_truths



def main():
    prompts, ground_truths, reward_kk = get_kk_test_prompts()
    #prompts, ground_truths = get_ob_test_prompts()
    #prompts, ground_truths = get_aime_prompts()
    #prompts, ground_truths = get_math_test_prompts()
    
    prompts, ground_truths = prompts[:100], ground_truths[:100]
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=8000
    )
    outputs = llm.generate(prompts, sampling_params)
    
    total_correctness = 0.0
    total_math_verify_score = 0.0
    total_length = 0
    num_samples = len(outputs)
    
    results = []
    
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[idx]
        #print(idx, gt)
        score_corr = reward_kk(generated_text, gt)
        #score_corr = 1 if reward_correct([generated_text], [gt])[0] == 2 else 0
        #math_verify_score = verify(parse(generated_text), parse(str(gt)))
        #print(parse(generated_text), parse(str(gt)), math_verify_score)
        #score_correct_and_format = reward_correct_and_format([generated_text], [gt])[0]
        token_length = len(llm.get_tokenizer().encode(generated_text))
        
        total_correctness += score_corr
        #total_math_verify_score += math_verify_score
        total_length += token_length
        
        results.append({
            "Prompt": prompts[idx],
            "Generated Answer": generated_text,
            "Ground Truth": gt,
            "Correctness_Score": score_corr,
           # "Math_verify_score": math_verify_score,
            #"Correctness_and_Format_Score": score_correct_and_format,
            "Response Length": token_length
        })
    
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    
    total_correctness = df["Correctness_Score"].sum()
    #total_math_verify_score = df["Math_verify_score"].sum()
    total_length = df["Response Length"].sum()
    
    #print(df["Mat"].value_counts())
    print(f"Average Correctness Score: {total_correctness / num_samples:.2f}")
    #print(f"Average Math Verify Score: {total_math_verify_score / num_samples:.2f}") 
    print(f"Average Response Length: {total_length / num_samples:.2f} tokens")

if __name__ == "__main__":
    main()
