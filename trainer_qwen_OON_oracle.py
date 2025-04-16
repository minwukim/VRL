import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from GRPO_customed.OON_GRPO_oracle  import OON_Oracle_GRPOTrainer

from datasets import load_dataset
from math_verify import verify, parse
from obsolete.custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from GRPO_customed.GRPO_custom import VerificationGRPOTrainer, SwitchingGRPOTrainer

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
    scale_reward: bool
    eval_on_start: bool
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False


from trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]
print(training_args)
# SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
# {prompt}
# Assistant: <think>"""
SYSTEM="""
<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>
"""

def reward_correct_a1_agnostic(completions, answer, **kwargs):

    # check if the strings ends with </think><answer>[boxed answer]</answer>
    def check_format_and_correctness(s, gt):
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
    
    return [check_format_and_correctness(c, gt) for c, gt in zip(completions, answer)]


def reward_correct_a1_dependent(completions, answer, first_completions=None, **kwargs):

    # Type 1: A2 accuracy > behavior collapse mitigation (i_c > c_c > i_i > c_i)
    i_c, c_c, i_i, c_i = 8, 0.5, -0.5, -1

    # Type 2: behavior collapse mitigation > A2 accuracy (i_c >= c_i > c_c >= i_i)
    # Note: for i->i, we might need to consider two cases and give rewards.
    # i_c, c_i, c_c, i_i, = 3, 0, -0.5, -1

    # check if the strings ends with </think><answer>[boxed answer]</answer>
    def check_format_and_correctness(s, gt):
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

    
    def give_a1_based_reward(a1, a2, gt, i_c=i_c, c_c=c_c, i_i=i_i, c_i=c_i):
        
        # If the format is wrong, return the value right away regardless of a1. 
        a2_score = check_format_and_correctness(a2,gt)
        if a2_score == -2:
            return -4
        if a2_score == -1:
            return -3

        a1_score = check_format_and_correctness(a1,gt)
        
        # Case 1: a1 is correct. 
        if a1_score == 2:
            # Case 1.1: correct to correct
            if a2_score == 2:
                return c_c
            # Case 1.2: correct to incorrect
            else:
                return c_i
            
        # Case 2: a1 is incorrect.
        else:
            # Case 2.1 incorrect to correct 
            if a2_score == 2:
                return i_c
            # Case 2.2 incorrect to incorrect
            else:
                return i_i

    # Case 1: For (Q->A1) setting
    if first_completions is None:
        return [check_format_and_correctness(c, gt) for c, gt in zip(completions, answer)]
    return [give_a1_based_reward(a1,a2,gt) for a1,a2,gt in zip (first_completions, completions, answer)]




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

model_path = training_args.model_name if not training_args.resume_from_checkpoint else training_args.checkpoint_path
model_name = AutoModelForCausalLM.from_pretrained(model_path)

grpo_config_args = GRPOConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
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
    #num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=True,
    vllm_max_model_len=training_args.vllm_max_model_len,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    #evaluation_strategy=training_args.evaluation_strategy,
    #eval_steps = training_args.eval_steps,
    #eval_on_start=training_args.eval_on_start,
)

trainer = OON_Oracle_GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct_a1_agnostic],
    # reward_funcs = [reward_correct_a1_dependent],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)
# trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else None)
trainer.train()
