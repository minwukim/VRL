import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
# from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

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


rewards = reward_func(responses, ground_truths)

df_rewards = pd.DataFrame(rewards, columns=["reward"])
counts = df_rewards["reward"].value_counts().sort_index()
total = len(rewards)
proportions = counts / total

reward_summary = pd.DataFrame({
    "Count": counts,
    "Proportion": proportions
})
print("\nReward Summary:")
print(reward_summary)




###############################################################################
# Continue with Confidence Score Collection and ECE Computation
###############################################################################

# Helper function to extract a confidence number (0-10) enclosed in \boxed{}.
def extract_confidence(text):
    pattern = r"\\boxed\s*{?\s*(\d+(?:\.\d+)?)\s*}?"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    else:
        return None

# Build new prompts for the confidence query with the required structure.
# The prompt now includes:
#  1. The system prompt (unchanged)
#  2. The original question (user role)
#  3. The assistant's first response (assistant role)
#  4. A new user instruction regarding the confidence interval.



confidence_prompts = []
for idx, example in enumerate(test_dataset):
    new_prompt = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': example["problem"]},
        {'role': 'assistant', 'content': responses[idx]},
        {'role': 'user', 'content': "How confident are you regarding your answer above? Please provide a single number from 0 to 10 (where 0 means no confidence and 10 means absolute certainty), enclosed within \\boxed{}. Answer only with the confidence level."}
    ]
    confidence_prompts.append(new_prompt)

print("confidence prompt example",confidence_prompts[0])
# Set sampling parameters for the confidence query (shorter output expected)
confidence_sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=50
)

# Run inference for the confidence prompts
confidence_outputs = llm.chat(confidence_prompts, confidence_sampling_params)
confidence_scores = []
for out in confidence_outputs:
    conf_text = out.outputs[0].text
    score = extract_confidence(conf_text)
    confidence_scores.append(score)

# For ECE calculation, treat a response as correct if its reward equals 1 and incorrect otherwise.
binary_accuracy = [1 if r == 1 else 0 for r in rewards]

# Normalize confidence scores from [0,10] to probabilities [0,1].
normalized_confidences = []
valid_indices = []  # store indices for which we extracted a valid confidence score
for i, conf in enumerate(confidence_scores):
    if conf is not None:
        normalized_confidences.append(conf / 10.0)
        valid_indices.append(i)

# Filter binary_accuracy for valid confidence scores.
binary_accuracy_valid = [binary_accuracy[i] for i in valid_indices]

# Compute Expected Calibration Error (ECE)
import numpy as np

num_bins = 10
bins = np.linspace(0.0, 1.0, num_bins + 1)

bin_acc = np.zeros(num_bins)
bin_conf = np.zeros(num_bins)
bin_counts = np.zeros(num_bins)

for conf, acc in zip(normalized_confidences, binary_accuracy_valid):
    bin_idx = min(np.digitize(conf, bins) - 1, num_bins - 1)
    bin_acc[bin_idx] += acc
    bin_conf[bin_idx] += conf
    bin_counts[bin_idx] += 1

ece = 0.0
total_valid = len(normalized_confidences)
for i in range(num_bins):
    if bin_counts[i] > 0:
        avg_acc = bin_acc[i] / bin_counts[i]
        avg_conf = bin_conf[i] / bin_counts[i]
        ece += (bin_counts[i] / total_valid) * abs(avg_acc - avg_conf)

print("\nConfidence Scores (raw):")
print(pd.DataFrame({"Confidence Score (raw)": confidence_scores}))
print("\nExpected Calibration Error (ECE): {:.4f}".format(ece))
