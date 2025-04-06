import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
# from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

##############################################
# Settings and Model Initialization
##############################################
# model_dir_path = ""
# checkpoint_num = "300"  # or update to your desired checkpoint number
# model_path = model_dir_path+"/checkpoint-"+checkpoint_num
# csv_file_path = "./evaluation_results"+model_dir_path+"checkpoint-"+checkpoint_num+".csv"

model_path = "Qwen/Qwen2.5-Math-1.5B-Instruct"

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


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
test_dataset = test_dataset.map(lambda x: {
    "prompt": [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x["problem"]}
    ],
    "answer": x["answer"],
    "level": x["level"]
})


def reward_func(completions,answer, **kwargs):
    def check_format_and_correctess(completion, ground_truth):
        response = last_boxed_only_string(completion)
        if response is None:
            return -1
        if verify(parse(response), parse(ground_truth)):   
            return -0.5
        return 1
    return [check_format_and_correctess(c, gt) for c, gt in zip(completions, answer)]


llm = LLM(model=model_path)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1500
)

prompts = []
ground_truths = []
responses = []

for example in test_dataset:
    prompts.append(example["prompt"])
    ground_truths.append(example["answer"])

outputs = llm.chat(prompts, sampling_params)

for idx, out in enumerate(outputs):
    response = out.outputs[0].text
    responses.append(response)

##############################################
# Compare Responses with Ground Truth using Reward Function
##############################################
rewards = reward_func(responses, ground_truths)

# Create a DataFrame to display counts and proportions of each reward value
df_rewards = pd.DataFrame(rewards, columns=["reward"])
counts = df_rewards["reward"].value_counts().sort_index()
total = len(rewards)
proportions = counts / total

# Print out the results
print("Reward Counts:")
print(counts)
print("\nReward Proportions:")
print(proportions)

# Alternatively, combine the counts and proportions into a single DataFrame for clearer presentation:
reward_summary = pd.DataFrame({
    "Count": counts,
    "Proportion": proportions
})
print("\nReward Summary:")
print(reward_summary)