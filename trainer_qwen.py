import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from datasets import load_dataset
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

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
    resume_from_checkpoint: bool = False


from trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]
print(training_args)

SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""


def reward_correct_and_format(completions, answer, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"</think>\n?<answer>([\s\S]*)</answer>", completion) for completion in completions] 
    completions = [match.group(1) if match else "" for match in matches]
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    correct_with_format = [1.0 if verify(parse(c), parse(gt))  else 0.0 for c, gt in zip(contents, answer)]

    return correct_with_format

def reward_correct(completions, answer, **kwargs):
    correct = [1.0 if verify(parse(c), parse(gt))  else 0.0 for c, gt in zip(completions, answer)]
    return correct

def extract_boxed_answer(solution):
    return last_boxed_only_string(solution)

def get_dataset():
    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": extract_boxed_answer(x["solution"]),
        "level": x["level"]
        })

    
    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["answer"],
        "level": x["level"]
        })
    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test

train, test = get_dataset()

model_name = training_args.model_name

grpo_config_args = GRPOConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
    resume_from_checkpoint=training_args.resume_from_checkpoint,
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
    #num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=training_args.use_vllm,
    vllm_max_model_len=training_args.vllm_max_model_len,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    #evaluation_strategy=training_args.evaluation_strategy,
    #eval_steps = training_args.eval_steps,
    #eval_on_start=training_args.eval_on_start,
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct, reward_correct_and_format],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)
trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
