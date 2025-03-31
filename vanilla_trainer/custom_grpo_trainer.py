from trl import GRPOConfig, GRPOTrainer

import os
import textwrap
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        GenerationConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        Trainer,
        TrainerCallback,
        is_wandb_available,
        )
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from profiling import profiling_context, profiling_decorator
from utils import pad

class NewGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
            ) -> dict[str, Union[torch.Tensor, Any]]:
        
         prompts = [x["prompt"] for x in inputs]
         
         device = self.accelerator.device
         self.num_iterations = 1
         prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
         prompt_inputs = self.processing_class(
                 text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                 )
         prompt_inputs = super()._prepare_inputs(prompt_inputs)
         prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

         if self.max_prompt_length is not None:
             prompt_ids = prompt_ids[:, -self.max_prompt_length :]
             prompt_mask = prompt_mask[:, -self.max_prompt_length :]

         # Generate completions using either vLLM or regular generation
         if self.args.use_vllm:
             # First, have main process load weights if needed
             if self.state.global_step != self._last_loaded_step:
                 self._move_model_to_vllm()
                 self._last_loaded_step = self.state.global_step

             # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
             all_prompts_text = gather_object(prompts_text)
             if self.accelerator.is_main_process:
                 # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                 # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                 # prompt individually.
                 ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                 with profiling_context(self, "vLLM.generate"):
                     all_outputs = self.llm.generate(
                             ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                             )
                     completion_ids = []
                 for outputs in all_outputs:
                     for output in outputs.outputs:
                         completion_ids.append(output.token_ids)
             else:
                 completion_ids = [None] * len(all_prompts_text)
             # Broadcast the completions from the main process to all processes, ensuring each process receives its
             # corresponding slice.
             completion_ids = broadcast_object_list(completion_ids, from_process=0)
             process_slice = slice(
                     self.accelerator.process_index * len(prompts),
                     (self.accelerator.process_index + 1) * len(prompts),
                     )
             completion_ids = completion_ids[process_slice]

             # Pad the completions, and concatenate them with the prompts
             completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
             completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
             prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
         else:
             # Regular generation path
             with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                 prompt_completion_ids = unwrapped_model.generate(
                         prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                         )

                 # Compute prompt length and extract completion ids
             prompt_length = prompt_ids.size(1)
             prompt_ids = prompt_completion_ids[:, :prompt_length]
             completion_ids = prompt_completion_ids[:, prompt_length:]

         # Mask everything after the first EOS token
         is_eos = completion_ids == self.processing_class.eos_token_id
         eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
         eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
         sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
         completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

         # Concatenate prompt_mask with completion_mask for logit computation
         attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

         logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

         with torch.no_grad():
             # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
             # computation here, and use per_token_logps.detach() instead.
             if self.num_iterations > 1:
                 old_per_token_logps = self._get_per_token_logps(
                         self.model, prompt_completion_ids, attention_mask, logits_to_keep
                         )
             else:
                 old_per_token_logps = None

             if self.beta == 0.0:
                 ref_per_token_logps = None
             elif self.ref_model is not None:
                 ref_per_token_logps = self._get_per_token_logps(
                         self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                         )
             else:
                 with self.accelerator.unwrap_model(self.model).disable_adapter():
                     ref_per_token_logps = self._get_per_token_logps(
                             self.model, prompt_completion_ids, attention_mask, logits_to_keep
                             )

                     # Decode the generated completions
         completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
         if is_conversational(inputs[0]):
             completions = []
             for prompt, completion in zip(prompts, completions_text):
                 bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                 completions.append([{"role": "assistant", "content": bootstrap + completion}])
         else:
             completions = completions_text

         rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
         for i, (reward_func, reward_processing_class) in enumerate(
                 zip(self.reward_funcs, self.reward_processing_classes)
                 ):
             if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                 reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
             else:
                 reward_func_name = reward_func.__name__
             with profiling_context(self, reward_func_name):
                 if isinstance(
                         reward_func, nn.Module
                         ):  # Module instead of PretrainedModel for compat with compiled models
                     if is_conversational(inputs[0]):
                         messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                         texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                     else:
                         texts = [p + c for p, c in zip(prompts, completions)]
                     reward_inputs = reward_processing_class(
                             text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                             )
                     reward_inputs = super()._prepare_inputs(reward_inputs)
                     with torch.inference_mode():
                         rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                 else:
                     # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                     keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                     reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                     output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                     # Convert None values to NaN
                     output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                     rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

         # If all reward functions return None for a given row, issue a detailed warning
         if torch.isnan(rewards_per_func).all(dim=1).any():
             nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
             row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
             row_reward_kwargs["prompt"] = prompts[nan_row_idx]
             row_reward_kwargs["completion"] = completions[nan_row_idx]
             warnings.warn(
                     f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                     "Please ensure that at least one reward function returns a valid reward."
                     )

             # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
         # completions may be distributed across processes
         rewards_per_func = gather(rewards_per_func)

         # Apply weights to each reward function's output and sum
         rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

         # Compute grouped-wise rewards
         mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
         std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

         # Normalize the rewards to compute the advantages
         mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
         std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
         advantages = (rewards - mean_grouped_rewards)

         #advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
         # Slice to keep only the local part of the data
         process_slice = slice(
                 self.accelerator.process_index * len(prompts),
                 (self.accelerator.process_index + 1) * len(prompts),
                 )
         advantages = advantages[process_slice]

         # Log the metrics
         mode = "eval" if self.control.should_evaluate else "train"

         completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
         self._metrics[mode]["completion_length"].append(completion_length)

         # Calculate mean reward per function, but only for samples where the function was applied
         for i, reward_func in enumerate(self.reward_funcs):
             if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                 reward_func_name = reward_func.config._name_or_path.split("/")[-1]
             else:
                 reward_func_name = reward_func.__name__
             # Only calculate mean for samples where this reward function was applied (non-NaN values)
             mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
             self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
         self._metrics[mode]["reward"].append(rewards.mean().item())
         self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

         if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
             prompts_to_log = gather_object(prompts_text)
             completions_to_log = gather_object(completions_text)
             rewards_to_log = rewards.tolist()

             if self.accelerator.is_main_process:
                 if is_rich_available():
                     print_prompt_completions_sample(
                             prompts_to_log,
                             completions_to_log,
                             rewards_to_log,
                             self.state.global_step,
                             )
                     if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                         import pandas as pd

                     # For logging
                     table = {
                             "step": [str(self.state.global_step)] * len(rewards),
                             "prompt": prompts_to_log,
                             "completion": completions_to_log,
                             "reward": rewards.tolist(),
                             }
                     df = pd.DataFrame(table)
                     wandb.log({"completions": wandb.Table(dataframe=df)})

         return {
                 "prompt_ids": prompt_ids,
                 "prompt_mask": prompt_mask,
                 "completion_ids": completion_ids,
                 "completion_mask": completion_mask,
                 "old_per_token_logps": old_per_token_logps,
                 "ref_per_token_logps": ref_per_token_logps,
                 "advantages": advantages,
                 }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
         if return_outputs:
             raise ValueError("The GRPOTrainer does not support returning outputs")
         # Compute the per-token log probabilities for the model
        
         prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
         completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
         

         self.num_iterations = 1
         self.epsilon=0.2
         input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
         attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
         logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

         per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

         # Compute the KL divergence between the model and the reference model
         if self.beta != 0.0:
             ref_per_token_logps = inputs["ref_per_token_logps"]
             per_token_kl = (
                     torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                     )

             # Compute the loss
         advantages = inputs["advantages"]
         # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
         # _generate_and_score_completions) and use per_token_logps.detach() instead.
         old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
         coef_1 = torch.exp(per_token_logps - old_per_token_logps)
         coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
         #coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
         per_token_loss1 = coef_1 * advantages.unsqueeze(1)
         per_token_loss2 = coef_2 * advantages.unsqueeze(1)
         per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
         if self.beta != 0.0:
             per_token_loss = per_token_loss + self.beta * per_token_kl
         loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

         # Log the metrics
         mode = "eval" if self.control.should_evaluate else "train"

         if self.beta != 0.0:
             mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
             self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

         is_clipped = (per_token_loss1 < per_token_loss2).float()
         clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
         self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
         return loss
