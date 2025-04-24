import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

from dataclasses import dataclass
from typing import Optional

from dataset_loader import load_math, load_countdown, load_kk
import pandas as pd

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


from trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]

#df = pd.read_csv("qwq_samples.csv")
df = pd.read_csv("self_distill_150_data_sep.csv")
# Combine 'question' and 'llm_answer' into a single 'text' column
df['text'] = df['prompt'] + df['response']

# Randomly sample 500 rows for the test set
# test_set = df.sample(n=100, random_state=42)

# # Remaining rows for the training set
# train_set = df.drop(test_set.index)

# Convert to Hugging Face Dataset format
# test_dataset = Dataset.from_pandas(test_set.reset_index(drop=True))
train_dataset = Dataset.from_pandas(df.reset_index(drop=True))


model_path = training_args.model_name if not training_args.resume_from_checkpoint else training_args.checkpoint_path
model_name = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side='left'

collator = DataCollatorForCompletionOnlyLM(
    response_template=tokenizer.encode(" [SEP] ", add_special_tokens=False),
    tokenizer=tokenizer,
)

sft_config_args = SFTConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
    resume_from_checkpoint=training_args.resume_from_checkpoint,
    adam_beta1=training_args.adam_beta1,
    adam_beta2=training_args.adam_beta2,
    weight_decay=training_args.weight_decay,
    warmup_steps=training_args.warmup_steps,
    lr_scheduler_type=training_args.lr_scheduler_type,
    logging_steps=training_args.logging_steps,
    bf16=training_args.bf16,
    # bf16_full_eval=training_args.bf16_full_eval,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    max_seq_length=training_args.max_completion_length,
    num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    max_steps=training_args.max_steps,
    eval_strategy="no",
    # eval_steps = training_args.eval_steps,
    # eval_on_start=training_args.eval_on_start,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # per_device_eval_batch_size=training_args.per_device_train_batch_size,
    # eval_accumulation_steps=training_args.gradient_accumulation_steps
)

trainer = SFTTrainer(
    model=model_name,
    args=sft_config_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=collator
)

## Get a sample from your dataset
#sample = train_dataset[0]
#
## Process it with your collator
#batch = collator([tokenizer.encode(sample['text'], truncation=True)])
#
## Examine the labels
#input_ids = batch["input_ids"][0]
#labels = batch["labels"][0]
#
## Print them side by side for comparison
#for i, (input_id, label) in enumerate(zip(input_ids, labels)):
#    token = tokenizer.decode([input_id])
#    print(f"Position {i}: Token '{token}' - Input ID: {input_id}, Label: {label}")


trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)
