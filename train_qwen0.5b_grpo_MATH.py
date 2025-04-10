import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from GRPO_customed.OON_GRPO import OON_GRPOTrainer
from GRPO_customed.ONN_GRPO import ONN_GRPOTrainer

from math_verify import verify, parse

max_seq_length = 1500
max_prompt_length = 1500 + 500

model_name = "Qwen/Qwen2.5-0.5B"

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

train_dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
# train_dataset = train_dataset.filter(lambda x: x['level'] not in ['Level 1', 'Level 2'])

test_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

train_dataset = train_dataset.map(lambda x: {
    "prompt": [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x["problem"]}
    ],
    "answer": last_boxed_only_string(x["solution"]),
    "level": x["level"]
})

test_dataset = test_dataset.map(lambda x: {
    "prompt": [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x["problem"]}
    ],
    "answer": x["answer"],
    "level": x["level"]
})

# def reward_func(completions,answer, **kwargs):
#     def check_format_and_correctess(completion, ground_truth):
#         response = last_boxed_only_string(completion)
#         if response is None:
#             return -1
#         if verify(parse(response), parse(ground_truth)):   
#             return 1
#         return -0.5
#     completions = [completion[0]['content'] for completion in completions]
#     return [check_format_and_correctess(c, gt) for c, gt in zip(completions, answer)]


def reward_func(completions,answer, **kwargs):
    def check_format_and_correctess(completion, ground_truth):
        if verify(parse(completion), parse(ground_truth)):   
            return 1
        return 0
    # completions = [completion[0]['content'] for completion in completions]
    return [check_format_and_correctess(c, gt) for c, gt in zip(completions, answer)]

training_args = GRPOConfig(
    use_vllm = True,
    output_dir = "0410-purerl-qwen0.5b",
    bf16 = True,
    bf16_full_eval=True,
    vllm_gpu_memory_utilization=0.9,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length,
    run_name = "0312-purerl-qwen0.5b-2",
    report_to = "wandb", 
    do_eval=True,
    per_device_train_batch_size=2,
    num_generations = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs = 8,
    logging_steps=1,
    gradient_checkpointing=True,
    save_strategy = "steps",
    save_steps = 200,
    eval_strategy="steps",
    eval_steps = 200,
    log_completions = True
    # eval_on_start=True,
)

# trainer = OON_GRPOTrainer(
trainer = ONN_GRPOTrainer(
    model=model_name,
    reward_funcs = [
        reward_func
    ],
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()