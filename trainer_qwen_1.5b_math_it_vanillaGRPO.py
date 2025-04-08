import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, TrlParser

from datasets import load_dataset
from math_verify import verify, parse

from dataclasses import dataclass
from typing import Optional

@dataclass
class MyArguments: 
    model_name: str
    output_dir: str
    run_name: str
    learning_rate: float
    beta: float
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    warmup_steps: int
    lr_scheduler_type: str
    logging_steps: float
    bf16: bool
    bf16_full_eval: bool
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    num_train_epochs: int
    save_steps: int
    max_grad_norm: float
    report_to: str
    use_vllm: bool
    vllm_max_model_len: int
    max_steps: int
    log_completions: bool
    evaluation_strategy: str
    eval_steps: int
    eval_on_start: bool
    scale_rewards: bool
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False



from trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]
print(training_args)


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
train_dataset = train_dataset.filter(lambda x: x['level'] not in ['Level 1', 'Level 2'])

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
    completions = [completion[0]['content'] for completion in completions]
    return [check_format_and_correctess(c, gt) for c, gt in zip(completions, answer)]


model_path = training_args.model_name if not training_args.resume_from_checkpoint else training_args.checkpoint_path
model_name = AutoModelForCausalLM.from_pretrained(model_path)

grpo_config_args = GRPOConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
    scale_rewards=training_args.scale_rewards,
    # resume_from_checkpoint=training_args.resume_from_checkpoint,
    beta=training_args.beta,
    adam_beta1=training_args.adam_beta1,
    adam_beta2=training_args.adam_beta2,
    weight_decay=training_args.weight_decay,
    warmup_steps=training_args.warmup_steps,
    lr_scheduler_type=training_args.lr_scheduler_type,
    logging_steps=training_args.logging_steps,
    bf16=training_args.bf16,
    bf16_full_eval=training_args.bf16_full_eval,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    num_generations=training_args.num_generations,
    max_prompt_length=training_args.max_prompt_length,
    max_completion_length=training_args.max_completion_length,
    # num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=True,
    vllm_max_model_len=training_args.vllm_max_model_len,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    # evaluation_strategy=training_args.evaluation_strategy,
    # eval_steps = training_args.eval_steps,
    # eval_on_start=training_args.eval_on_start,
)

trainer = GRPOTrainer(
    model=model_name,
    # reward_funcs=[reward_correct_a1_agnostic],
    reward_funcs = [reward_func],
    args=grpo_config_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)
trainer.train()
