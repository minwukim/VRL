import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trl import GRPOConfig, GRPOTrainer

from datasets import load_dataset
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

from dataclasses import dataclass
from typing import Optional

from dataset_loader import load_math, load_countdown, load_kk, load_mmlupro

#from custom_grpo_trainer import NewGRPOTrainer

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
    log_completions: bool
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False
    max_steps: int = -1
    eval_on_start: bool = False
    eval_steps: int = None
    evaluation_strategy: str = None


from trl.trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]
print(training_args)

SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

#def reward_think_answer_format(completions, answer, **kwargs):
#    # check if the strings ends with </think><answer>...</answer>
#    def check_format(s):
#        pattern = r".+</think>\s*<answer>.+</answer>\s*$"
#        if s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1:
#            if bool(re.fullmatch(pattern, s, re.DOTALL)): return 0
#        return -1
#
#    return [check_format(completion) for completion in completions]
#
#def reward_boxed_format(completions, answer, **kwargs):
#    # check if the strings ends with </think><answer>[boxed answer]</answer>
#    def check_format(s):
#        pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
#        if s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1:
#            match = re.search(pattern, s, re.DOTALL)
#            if not match: return 0
#            else:
#                if last_boxed_only_string(match.group(1)): return 0
#                else: return -0.5
#        else: return 0
#
#    return [check_format(completion) for completion in completions]


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


#train, test = get_dataset()
#train, test = load_math(SYSTEM, sample=100)
#train, test, reward_correct = load_kk()


# humanities = ["history","law","philosophy"]
humanities = ["physics"]
train, test, reward_correct = load_mmlupro(sample=0.75, categories=humanities)

model_path = training_args.model_name if not training_args.resume_from_checkpoint else training_args.checkpoint_path
model_name = AutoModelForCausalLM.from_pretrained(model_path)

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
    num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=training_args.use_vllm,
    vllm_max_model_len=training_args.vllm_max_model_len,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    evaluation_strategy="no",
    temperature=0.7,
    top_p=0.95
    # vllm_gpu_memory_utilization=0.7,
    #evaluation_strategy=training_args.evaluation_strategy,
    #eval_steps = training_args.eval_steps,
    #eval_on_start=training_args.eval_on_start,
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)
trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)

