import pandas as pd
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

# 1. load the MATH dataset
print("Loading MATH dataset...")
math_dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
questions = [item['problem'] for item in math_dataset.select(range(10))]

# 2. Load the model with vLLM
print("Loading model with vLLM...")
llm = LLM(model="Qwen/Qwen-2.5-Math-1.5B", tensor_parallel_size=1, max_model_len=4000)

# 3. Define the sampling parameters
temperature = 0.9
top_p = 1
top_k = 50
n = 1

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_tokens=4000,
    n=n,
)

# 4. Prepare prompts
def prepare_prompt(question):
    return f"{question}"

prompts = [prepare_prompt(q) for q in questions]

# 5. Generate responses
print("Generating responses...")
outputs = llm.generate(prompts, sampling_params)

# 6. Collect responses and prepare for CSV export.
results_list = []

def calculate_entropy(logprobs_dict):
    """Calculates the entropy from a dictionary of log probabilities."""
    if not logprobs_dict:
        return 0.0
    # Convert log probabilities to probabilities
    probs = [torch.exp(torch.tensor(logp.logprob)) for logp in logprobs_dict.values()]
    probs_tensor = torch.tensor(probs)
    # Normalize the probabilities to sum to 1
    # This is an approximation if we only have top-k logprobs
    # For a more accurate entropy, the full distribution is needed.
    normalized_probs = probs_tensor / probs_tensor.sum()
    entropy = -torch.sum(normalized_probs * torch.log(normalized_probs))
    return entropy.item()

print("Processing and saving results...")

for i, output in enumerate(outputs):
    prompt_text = output.prompt
    for j, generated_sequence in enumerate(output.outputs):
        sequence_text = generated_sequence.text
        token_ids = generated_sequence.token_ids
        # This will be a list of dictionaries, one for each token
        sequence_logprobs = generated_sequence.logprobs

        if sequence_logprobs:
            # Extract logit values for each token in the sequence
            # Note: vLLM provides log probabilities, not raw logits.
            # The logit is proportional to the log probability.
            token_logprobs = [step[token_ids[k]].logprob if k < len(token_ids) and step and token_ids[k] in step else None for k, step in enumerate(sequence_logprobs)]
            # Calculate entropy for each token's probability distribution
            token_entropies = [calculate_entropy(step) if step else 0.0 for step in sequence_logprobs]
        else:
            token_logprobs = []
            token_entropies = []

        results_list.append({
            "question_index": i,
            "question": questions[i],
            "generation_index": j,
            "generated_text": sequence_text,
            "token_ids": token_ids,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies
        })

# Create a DataFrame and save to CSV
df = pd.DataFrame(results_list)
df.to_csv("qwq_32b_experiments.csv", index=False)
print("Experiment results saved to qwq_32b_experiments.csv")


