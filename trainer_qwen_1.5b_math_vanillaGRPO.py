import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, TrlParser

from datasets import load_dataset
from math_verify import verify, parse
from custom_MATH_reward import last_boxed_only_string
# from GRPO_custom import VerificationGRPOTrainer, SwitchingGRPOTrainer

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
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False


from trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]
print(training_args)


SYSTEM = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks through the reasoning process, then provides the User with the answer. "
    "The Assistant's response includes three parts: the reasoning process, the final answer, and a confidence score (from 0 to 10). "
    "Each part is enclosed in specific tags as follows:\n\n"
    "<think> reasoning process here </think>\n"
    "<answer>\\boxed{{final answer here}}</answer>\n"
    "<confidence>\\boxed{{confidence score here}}</confidence>\n\n"
    "The confidence score must be an integer between 0 and 10, inclusive.\n"
    "The final answer and confidence score will be extracted automatically from the \\boxed{{}} tags.\n"
    "User: Help me solve the math question: {prompt}\n"
    "Assistant: <think>"
)



def reward_func(completions, answer, **kwargs):
    """
    Evaluate each completion by checking if it follows the specified format and 
    whether it matches the ground-truth answers. Returns a numeric reward for each completion.

    Format requirements:
    1. Exactly one occurrence each of:
       </think>, <answer>, </answer>, <confidence>, </confidence>
    2. Completion must match the regex pattern that extracts <answer>... and <confidence>...
    3. <answer> and <confidence> content must each contain a valid \\boxed{} expression.
    4. <confidence> must be a digit between 0 and 10 (inclusive is optional to enforce).
    5. If the answer inside <answer> matches the ground-truth after parsing, reward = 2. 
       Otherwise, reward = -0.5.
    6. If the format is incorrect, reward = -2. 
       If the format is okay but missing required \\boxed{} tags or confidence is invalid, reward = -1.

    Args:
        completions (list[str]): The generated completions to evaluate.
        answers (list[str]): The corresponding ground-truth answers.
        **kwargs: Extra arguments (currently unused).

    Returns:
        list[float]: A list of reward scores for each completion.
    """

    def check_format_and_correctness(completion, ground_truth):
        # Regex pattern to match the text within <answer>...</answer> and <confidence>...</confidence>
        pattern = r".+</think>\s*<answer>(.+)</answer>\s*<confidence>(.+)</confidence>\s*$"

        # Verify the exact count of required tags. If not exactly one of each, return -2.
        if not (
            completion.count("</think>") == 1
            and completion.count("<answer>") == 1
            and completion.count("</answer>") == 1
            and completion.count("<confidence>") == 1
            and completion.count("</confidence>") == 1
        ):
            return -2  # Missing or extra format tokens.

        # Check if the completion matches the regex structure.
        match = re.search(pattern, completion, re.DOTALL)
        if not match:
            return -2  # Does not fit the required pattern.

        # Extract text within <answer> and <confidence>.
        extracted_response = last_boxed_only_string(match.group(1))
        extracted_confidence = last_boxed_only_string(match.group(2))

        if extracted_response is None:
            return -1  # Missing valid \boxed{} in <answer>.
        if extracted_confidence is None:
            return -1  # Missing valid \boxed{} in <confidence>.

        # Remove wrapping from \boxed{} in the confidence string and validate it's numeric.
        extracted_confidence = extracted_confidence.replace("\\boxed{", "").replace("}", "")
        # Check if it is a valid integer.
        if not extracted_confidence.isdigit():
            return -1  
        # Check if the confidence value is a digit between 0 and 10.
        if not (0 <= int(extracted_confidence) <= 10):
            return -1
        
        # (Optional) Check confidence range, e.g., 0 <= int(extracted_confidence) <= 10
        # confidence_val = int(extracted_confidence)

        # Compare the parsed answer to the ground-truth. 
        # 'verify' and 'parse' are user-defined functions that check correctness.
        if verify(parse(extracted_response), parse(ground_truth)):
            return 2   # Correct answer with valid format.
        else:
            return -0.5  # Format is fine but the answer is incorrect.

    # Apply the check to each (completion, answer) pair.
    return [check_format_and_correctness(c, gt) for c, gt in zip(completions, answer)]




def get_dataset():
    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": last_boxed_only_string(x["solution"]),
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

trainer = GRPOTrainer(
    model=model_name,
    # reward_funcs=[reward_correct_a1_agnostic],
    reward_funcs = [reward_func],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)
# trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)
trainer.train()
