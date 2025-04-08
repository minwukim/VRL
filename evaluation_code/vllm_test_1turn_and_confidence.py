import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_verify import verify, parse


# model_path = "Qwen/Qwen2.5-Math-1.5B-Instruct"
# csv_path = "qwen_1.5b_math_it_base.csv"

model_path = "../qwen2.5-1.5B-MATH-it-vanilla-GRPO-scaleTrue/checkpoint-300"
csv_path = "qwen_1.5b_math_it_vanilla_scaleTrue_cp300.csv"



##############################################
# Helper Functions
##############################################
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

# Helper function to extract a confidence number (0-10) enclosed in \boxed{}.
def extract_confidence(text):
    pattern = r"\\boxed\s*{?\s*(\d+(?:\.\d+)?)\s*}?"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    else:
        return None

##############################################
# Settings and Model Initialization
##############################################

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Load and process the test dataset
test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
test_dataset = test_dataset.map(lambda x: {
    "prompt": [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x["problem"]}
    ],
    "answer": x["answer"],
    "level": x["level"],
    "problem": x["problem"]  # store original problem for later reference
})

# Reward function: returns -1 if formatting is missing, -0.5 if correct, and 1 otherwise.
def reward_func(completions, answer, **kwargs):
    def check_format_and_correctess(completion, ground_truth):
        if verify(parse(completion), parse(ground_truth)):   
            return 1
        return 0
    return [check_format_and_correctess(c, gt) for c, gt in zip(completions, answer)]

# Initialize the model
llm = LLM(model=model_path)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1800
)

##############################################
# First Round: Generate Answers
##############################################
prompts = []
ground_truths = []
responses = []

for example in test_dataset:
    prompts.append(example["prompt"])
    ground_truths.append(example["answer"])

# Generate first responses
outputs = llm.chat(prompts, sampling_params)
for out in outputs:
    responses.append(out.outputs[0].text)

# Compute rewards for the first responses
rewards = reward_func(responses, ground_truths)

# For ECE, binary correctness: per your instructions, treat a response as correct if reward equals 1.
binary_accuracy = [1 if r == 1 else 0 for r in rewards]

# Aggregate reward counts and proportions (optional print-out)
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

##############################################
# Second Round: Collect Confidence Scores
##############################################
confidence_prompts = []
for idx, example in enumerate(test_dataset):
    # Build new prompt including system, user (problem), assistant (first response), and a new user query for confidence.
    new_prompt = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': example["problem"]},
        {'role': 'assistant', 'content': responses[idx]},
        {'role': 'user', 'content': "How confident are you regarding your answer above? Please provide a single number from 0 to 10 (where 0 means no confidence and 10 means absolute certainty), enclosed within \\boxed{}. Answer only with the confidence level."}
    ]
    confidence_prompts.append(new_prompt)

# (Optional) print an example confidence prompt for inspection
print("Confidence prompt example:", confidence_prompts[0])

# Set sampling parameters for confidence query (expecting shorter output)
confidence_sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=50
)

# Generate confidence responses
confidence_outputs = llm.chat(confidence_prompts, confidence_sampling_params)
confidence_responses = []
confidence_scores = []
for out in confidence_outputs:
    conf_text = out.outputs[0].text
    confidence_responses.append(conf_text)
    score = extract_confidence(conf_text)
    confidence_scores.append(score)

##############################################
# Expected Calibration Error (ECE) Computation
##############################################
# Normalize confidence scores from [0,10] to probabilities [0,1].
normalized_confidences = []
valid_indices = []  # store indices for which we extracted a valid confidence score
for i, conf in enumerate(confidence_scores):
    if conf is not None:
        normalized_confidences.append(conf / 10.0)
        valid_indices.append(i)

# Filter binary_accuracy for valid confidence scores.
binary_accuracy_valid = [binary_accuracy[i] for i in valid_indices]

# Compute ECE with 10 bins.
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

##############################################
# Save Detailed Results to CSV
##############################################
results = []
for i, example in enumerate(test_dataset):
    # Prepare a string representation of the first prompt
    # (Concatenate role and content for clarity.)
    first_prompt_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in example["prompt"]])
    
    # Define correctness: per your ECE definition, a response is "correct" if reward equals 1.
    correctness = 1 if rewards[i] == 1 else 0
    
    results.append({
        "Question": example["problem"],
        "Ground Truth": example["answer"],
        "First Prompt": first_prompt_str,
        "First Response": responses[i],
        "Reward": rewards[i],
        "Correctness": correctness,
        "Second Response": confidence_responses[i],
        "Extracted Confidence Score": confidence_scores[i]
    })

results_df = pd.DataFrame(results)
# Save the results DataFrame to a CSV file.
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")
