import re
from datasets import load_dataset, Dataset
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from trl import GRPOConfig, GRPOTrainer

max_seq_length = 2500
max_prompt_length = max_seq_length + 256
model_name = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """
Answer the question below in the specified format. 
First, carefully think through the reasoning process. Then, provide a refined solution explaining the steps of the solution. 
Enclose the reasoning process within <think> </think> tags and the final solution within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> solution here </answer>. 
Ensure the final answer in the solution is formatted within \\boxed{}.
"""

def extract_ground_truth(text: str) -> str | None:
    return remove_boxed(last_boxed_only_string(text))


def get_math_data(split: str) -> Dataset:
    if split == "train":
        source_name = "DigitalLearningGmbH/MATH-lighteval"
        data = load_dataset(source_name, trust_remote_code=True)['train']
    elif split == "test":
        source_name = "HuggingFaceH4/MATH-500"
        data = load_dataset(source_name, trust_remote_code=True)['test']
    else:
        raise ValueError("Invalid split. Choose either 'train' or 'test'.")

    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': extract_ground_truth(x['solution'])
    })

    return data

dataset_train = get_math_data("train")
dataset_test = get_math_data("test")

# check correctness: 1 if correct, 0 otherwise (incorrect or none)
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    scores = [compute_score(r, a) for r, a in zip(responses, answer)]

    # for i, (q, r, a, score) in enumerate(zip(prompts, responses, answer, scores)):
    #     print('-'*20, f"Question {i+1}:\n{q[-1]['content']}", 
    #           f"\nAnswer:\n{a}", 
    #           f"\nResponse:\n{responses[i]}", 
    #           f"\nExtracted:\n{r}", 
    #           f"\nScore:\n{score}")

    return scores

# check token format: If matches the regex ^<think>.*?</think><answer>.*?</answer>$, give 0.1, 0 otherwise. | Ref: https://huggingface.co/docs/trl/main/en/grpo_trainer
def token_format_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.1 if match else 0.0 for match in matches]

# check box format: if the answer is encased in \boxed, give 0.1. 0 otherwise. |Ref: https://huggingface.co/docs/trl/main/en/grpo_trainer
def boxed_format_reward_func(completions, **kwargs):
    # Regular expression to capture content inside \boxed{}
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(r"\\boxed\{(.*?)\}", r) for r in responses]
    return [0.1 if match else 0.0 for match in matches]


training_args = GRPOConfig(
    use_vllm = True,
    output_dir = "qwen3b-math-grpo",
    bf16 = True,
    bf16_full_eval=True,
    vllm_gpu_memory_utilization=0.6,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length,
    run_name = "qwen3b-grpo-exp2",
    report_to = "wandb", 
    do_eval=True,
    per_device_train_batch_size=4,
    num_generations = 4,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    logging_steps=1,
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs = [
        correctness_reward_func,
        token_format_reward_func,
        boxed_format_reward_func
    ],
    args = training_args,
    train_dataset=dataset_train,
)
trainer.train()