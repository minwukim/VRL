from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, TrlParser
from math_verify import verify, parse
from obsolete.custom_MATH_reward import last_boxed_only_string
import pandas as pd


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
    lr_scheduler_type: str
    logging_steps: float
    bf16: bool
    bf16_full_eval: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
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
    scale_reward: bool
    epsilon: float
    epsilon_high: float
    mask_truncated_completions: bool
    loss_type: str
    nsr_enabled: bool
    all_400: bool
    num_iterations: int
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False

parser = TrlParser(dataclass_types=[MyArguments])
training_args = parser.parse_args_and_config()[0]
print(training_args)

def reward_func(completions, answer, **kwargs):
    def reward(s, gt):
        # add the last boxed tag
        last_boxed = last_boxed_only_string(s)
        if last_boxed is not None:
            s = last_boxed
        try:
            is_correct = verify(parse(s), parse(gt))
            return 1 if is_correct else -1
        except:
            return -1  # parsing/verification failed
    return [reward(c, gt) for c, gt in zip(completions, answer)]


def get_dataset(all_400=True):

    df = pd.read_csv("1.5B_math_train_hit.csv")
    if all_400:
        df = df[df["hit"] <= 4]
    else:
        df = df[df['question_index'] == 2464]
    df = df[["prompt", "solution"]].copy()
    df["answer"] = df["solution"].apply(last_boxed_only_string)
    df = df.drop(columns=["solution"])
    train = Dataset.from_pandas(df, preserve_index=False)

    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test = test.map(lambda x: {
        "prompt": x["problem"],
        "answer": x["answer"],
        "level": x["level"]
    })
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])

    return train, test


train, test = get_dataset(training_args.all_400)

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
    lr_scheduler_type=training_args.lr_scheduler_type,
    logging_steps=training_args.logging_steps,
    bf16=training_args.bf16,
    bf16_full_eval=training_args.bf16_full_eval,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    num_generations=training_args.num_generations,
    max_prompt_length=training_args.max_prompt_length,
    max_completion_length=training_args.max_completion_length,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=True,
    vllm_max_model_len=training_args.vllm_max_model_len,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    eval_strategy=training_args.evaluation_strategy,
    eval_steps = training_args.eval_steps,
    eval_on_start=training_args.eval_on_start,
    epsilon=training_args.epsilon,
    epsilon_high=training_args.epsilon_high,
    mask_truncated_completions=training_args.mask_truncated_completions,
    loss_type=training_args.loss_type,
    nsr_enabled=training_args.nsr_enabled,
    num_iterations=training_args.num_iterations
)


trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_func],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)
# trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)
# trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else None)
trainer.train()


    