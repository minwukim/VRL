from typing import Callable, Optional, Union, Any, List, Dict

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.trainer.utils import pad
from copy import deepcopy

# Define RewardFunc as before
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class CustomGRPOTrainer(GRPOTrainer):
    """
    Custom GRPOTrainer with modified generation:
      1. For each unique question Q (after removing duplicates), generate one initial answer A.
      2. Then replicate (Q, A) pairs by num_generations times.
      3. Construct new prompts using the duplicated (Q, A) pairs and an added instruction.
      4. Generate judgment J from these identical prompts.
    This ensures that all duplicates of a prompt Q share the same initial answer A,
    which is required for correct grouping and advantage computation in GRPO.
    """
    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        
        # ------------------------------
        # Step 1: Prepare Original Prompts
        # ------------------------------
        # Extract original questions from each input sample.
        prompts = [x["prompt"] for x in inputs]  # List of questions Q
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        
        # Convert questions to token IDs
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        #### MINWU Q: Why not prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # ------------------------------
        # Step 1.1: Remove Duplicate Prompts
        # ------------------------------
        # Build a mapping from unique prompt text to indices in the original list.
        unique_prompts: Dict[str, List[int]] = {}
        for idx, p in enumerate(prompts_text):
            unique_prompts.setdefault(p, []).append(idx)
        
        unique_prompts_text = list(unique_prompts.keys())
        
        # ------------------------------
        # Step 2: Generate Initial Answer A for Unique Prompts
        # ------------------------------
        # Ensure the vLLM weights are reloaded if needed.
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate initial answer for each unique prompt.
        if self.args.use_vllm:
            print("YOU SHOULD SEE ME: using vllm to generate initial answers.")

            # Create a copy of sampling_params with n set to 1.
            single_sampling_params = deepcopy(self.sampling_params)
            single_sampling_params.n = 1
            
            all_unique_prompts = gather_object(unique_prompts_text)
            if self.accelerator.is_main_process:
                initial_outputs = self.llm.generate(
                    all_unique_prompts,
                    sampling_params=single_sampling_params,
                    use_tqdm=False,
                )

                unique_initial_answer_ids = []
                for outputs in initial_outputs:
                    # Taking the first generation from vLLM output for each unique prompt.
                    unique_initial_answer_ids.append(
                        torch.tensor(outputs.outputs[0].token_ids, device=device)
                    )
            else:
                unique_initial_answer_ids = [None] * len(unique_prompts_text)
            unique_initial_answer_ids = broadcast_object_list(unique_initial_answer_ids, from_process=0)
            # Pad the list of generated token IDs into a tensor.
            unique_initial_answer_ids = pad(unique_initial_answer_ids, padding_value=self.processing_class.pad_token_id)
        else:
            print("YOU SHOULD NOT SEE ME: using model to generate initial answers.")
            with torch.no_grad():
                unique_prompt_inputs = self.processing_class(
                    unique_prompts_text,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                unique_prompt_inputs = Trainer._prepare_inputs(self, unique_prompt_inputs)
                unique_prompt_ids = unique_prompt_inputs["input_ids"]
                unique_prompt_mask = unique_prompt_inputs["attention_mask"]
                if self.max_prompt_length is not None:
                    unique_prompt_ids = unique_prompt_ids[:, -self.max_prompt_length :]
                    unique_prompt_mask = unique_prompt_mask[:, -self.max_prompt_length :]
                generated = self.model.generate(
                    unique_prompt_ids,
                    attention_mask=unique_prompt_mask,
                    max_new_tokens=self.max_completion_length,
                    generation_config=self.generation_config,
                )
                # Keep only the answer part.
                unique_initial_answer_ids = generated[:, unique_prompt_ids.shape[1]:]
        
        # Decode generated answers to text
        unique_answers_text = self.processing_class.batch_decode(unique_initial_answer_ids, skip_special_tokens=True)
        print(f"Generated {len(unique_answers_text)} initial answers.")
        print(unique_answers_text[0])
        
        # Map the unique answers back to the original prompt order.
        answers_text = [None] * len(prompts_text)
        for prompt_text, indices in unique_prompts.items():
            # All instances of the same prompt receive the same answer.
            unique_idx = unique_prompts_text.index(prompt_text)
            answer = unique_answers_text[unique_idx]
            for idx in indices:
                answers_text[idx] = answer
        
        # ------------------------------
        # Step 2.1: Duplicate (Q,A) Pair by num_generations
        # ------------------------------
        # GRPO expects each prompt to be repeated num_generations times.
        # We duplicate the prompt text and its corresponding answer accordingly.
        duplicated_prompts_text = []
        duplicated_answers_text = []
        for q, a in zip(prompts_text, answers_text):
            # for _ in range(self.num_generations):
            for _ in range(1):
                duplicated_prompts_text.append(q)
                duplicated_answers_text.append(a)
        
        print(f"Generated {len(duplicated_prompts_text)} duplicated prompts.")
        # ------------------------------
        # Step 3: Construct New Prompts for Judgment
        # ------------------------------
        # Build new prompts by combining duplicated question and answer with an added instruction.
        added_instruction = (
            "\n Given the question and the response provided, go through the reasoning process of the response and check if the response is correct or not. "
            "Then, try to resolve the problem if incorrect, and return the same final answer if you think it is correct. "
            "Enclose your reasoning of checking and resolving process within <think> </think> tags and the final solution within <answer> </answer> tags, "
            "i.e., <think> reasoning process here </think> <answer> solution here </answer>. "
            "Ensure the final answer in the solution is formatted within \\boxed{}, so that the response can be directly extracted for grading."
        )

        new_prompts_text = []
        for q, a in zip(duplicated_prompts_text, duplicated_answers_text):
            new_prompt = f"Question: {q}\nAnswer: {a}{added_instruction}"
            new_prompts_text.append(new_prompt)
        
        # Tokenize the new prompts.
        new_prompt_inputs = self.processing_class(
            new_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        new_prompt_inputs = Trainer._prepare_inputs(self, new_prompt_inputs)
        new_prompt_ids, new_prompt_mask = new_prompt_inputs["input_ids"], new_prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            new_prompt_ids = new_prompt_ids[:, -self.max_prompt_length :]
            new_prompt_mask = new_prompt_mask[:, -self.max_prompt_length :]
        
        # ------------------------------
        # Step 4: Generate Judgment J
        # ------------------------------
        # Generate the judgment (expected to be "correct" or "incorrect")
        if self.args.use_vllm:
            all_new_prompts = gather_object(new_prompts_text)
            if self.accelerator.is_main_process:
                judgment_outputs = self.llm.generate(
                    all_new_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                judgment_ids_list = []
                for outputs in judgment_outputs:
                    judgment_ids_list.append(
                        torch.tensor(outputs.outputs[0].token_ids, device=device)
                    )
            else:
                judgment_ids_list = [None] * len(new_prompts_text)
            judgment_ids_list = broadcast_object_list(judgment_ids_list, from_process=0)
            judgment_ids = pad(judgment_ids_list, padding_value=self.processing_class.pad_token_id)
        else:
            with torch.no_grad():
                judgment_ids = self.model.generate(
                    new_prompt_ids,
                    attention_mask=new_prompt_mask,
                    max_new_tokens=self.max_completion_length,  # adjust if judgment is expected to be short
                    generation_config=self.generation_config,
                )
                judgment_ids = judgment_ids[:, new_prompt_ids.shape[1]:]
        
        judgments_text = self.processing_class.batch_decode(judgment_ids, skip_special_tokens=True)
        completion_mask = (judgment_ids != self.processing_class.pad_token_id).long()
        print(f"Generated {len(judgments_text)} judgments.")
        print(judgments_text[0])
        
        # ------------------------------
        # Step 5: Prepare for Log Probability Computation & Reward Calculation
        # ------------------------------
        # Concatenate the original question (from first stage) with the judgment tokens.
        prompt_completion_ids = torch.cat([prompt_ids, judgment_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = judgment_ids.size(1)
        
        with torch.no_grad():
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
        
        # ------------------------------
        # Step 6: Compute Rewards & Advantages Based on Judgment
        # ------------------------------
        rewards_per_func = torch.zeros(len(duplicated_prompts_text), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # If reward_func is a module, prepare its input via its processing class.
            if isinstance(reward_func, nn.Module):
                # Here, we assume that self.reward_processing_classes has been set up.
                reward_processing_class = self.reward_processing_classes[i]
                # If working with conversational data, you could wrap the texts accordingly.
                # For simplicity, we just concatenate the duplicated prompt and judgment.
                texts = [q + j for q, j in zip(duplicated_prompts_text, judgments_text)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # For non-module reward functions, pass along any extra keys if needed.
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward = reward_func(prompts=duplicated_prompts_text, completions=judgments_text, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward, dtype=torch.float32, device=device)
        
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        process_slice = slice(
            self.accelerator.process_index * len(duplicated_prompts_text),
            (self.accelerator.process_index + 1) * len(duplicated_prompts_text),
        )
        advantages = advantages[process_slice]

        
        # ------------------------------
        # Step 7: Logging Metrics (Optional)
        # ------------------------------
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            judgments_to_log = gather_object(judgments_text)
            rewards_to_log = rewards.tolist()
            if self.accelerator.is_main_process:
                if hasattr(self, "print_func") and callable(self.print_func):
                    self.print_func(prompts_to_log, judgments_to_log, rewards_to_log, self.state.global_step)
                try:
                    import wandb
                    import pandas as pd
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "judgment": judgments_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    print(df)
                    wandb.log({"completions": wandb.Table(dataframe=df)})
                except Exception:
                    print("YOU SHOULDN'T SEE ME: Failed to log completions.")
                    pass
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": judgment_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
