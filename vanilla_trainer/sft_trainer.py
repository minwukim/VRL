import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from dataclasses import dataclass

@dataclass
class MyArguments:
    model_name: str = "Qwen/Qwen2.5-3B"           # Replace with your model
    output_dir: str = "./sft_output"
    run_name: str = "sft_run"
    learning_rate: float = 2e-5
    beta: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    bf16: bool = False
    bf16_full_eval: bool = False
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    num_generations: int = 1
    max_prompt_length: int = 512
    max_completion_length: int = 256
    num_train_epochs: int = 2
    save_steps: int = 100
    max_grad_norm: float = 1.0
    report_to: str = "all"
    use_vllm: bool = False
    vllm_max_model_len: int = 2048
    log_completions: bool = False
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False
    max_steps: int = -1
    eval_on_start: bool = False
    eval_steps: int = None
    evaluation_strategy: str = None

# Instantiating the arguments (could also parse from CLI or YAML)
training_args = MyArguments()

# Load your CSV with 'prompt' and 'response' columns
df = pd.read_csv("self_distill_base_data.csv")

df['text'] = df['prompt'] + df['response']

print("Number of training examples:", len(df))
print("Sample row:\n", df.iloc[0])

# Use all data for training
train_dataset = Dataset.from_pandas(df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(training_args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # For most LLMs

model = AutoModelForCausalLM.from_pretrained(training_args.model_name, attn_implementation="flash_attention_2")

def custom_collate(samples):
    print("Sample keys:", samples[0].keys())
    print("Sample example:", samples[0])
    input_ids = []
    labels = []
    for s in samples:
        q_ids = tokenizer.encode(s['prompt'], add_special_tokens=False)
        a_ids = tokenizer.encode(s['response'], add_special_tokens=False)
        ids = q_ids + a_ids
        input_ids.append(ids)
        label = [-100]*len(q_ids) + a_ids
        labels.append(label)
    
    # Pad to max length in the batch
    max_len = max(len(ids) for ids in input_ids)
    input_ids = [ids + [tokenizer.pad_token_id]*(max_len - len(ids)) for ids in input_ids]
    labels = [lbl + [-100]*(max_len - len(lbl)) for lbl in labels]
    attention_mask = [[1]*len(ids) + [0]*(max_len - len(ids)) for ids in input_ids]
    
    batch = {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),
        'attention_mask': torch.tensor(attention_mask),
    }
    return batch

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
    bf16_full_eval=training_args.bf16_full_eval,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    max_seq_length=training_args.max_completion_length,
    num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    max_steps=training_args.max_steps,
    # No eval strategy, eval_steps, eval_on_start, etc.
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_eval_batch_size=training_args.per_device_train_batch_size,
    eval_accumulation_steps=training_args.gradient_accumulation_steps
)

train_samples = df[['prompt', 'response']].to_dict(orient='records')
trainer = SFTTrainer(
    model=model,
    args=sft_config_args,
    train_dataset=train_samples,
    eval_dataset=None,  # No test set
    data_collator=custom_collate,
)

trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)
