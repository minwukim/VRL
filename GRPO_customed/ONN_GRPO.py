from typing import Callable, Optional, Union, Any, List, Dict, Generator
import contextlib

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template, is_conversational
from trl.trainer.utils import pad
from copy import deepcopy

import pandas as pd

from transformers import Trainer

import wandb

import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import re
import functools

from copy import deepcopy

@contextlib.contextmanager
def profiling_context(trainer: Trainer, name: str) -> Generator[None, None, None]:
    """
    A context manager function for profiling a block of code. Results are logged to Weights & Biases if enabled.

    Args:
        trainer (`~transformers.Trainer`):
            Trainer object.
        name (`str`):
            Name of the block to be profiled. Used as a key in the logged dictionary.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_context

    class MyTrainer(Trainer):
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            with profiling_context(self, "matrix_multiplication"):
                # Code to profile: simulate a computationally expensive operation
                result = A @ B  # Matrix multiplication
    ```
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if "wandb" in trainer.args.report_to and wandb.run is not None and trainer.accelerator.is_main_process:
        wandb.log({f"profiling/Time taken: {trainer.__class__.__name__}.{name}": duration})



def print_prompt_completions_sample(prompts: list[str], completions: list[str], rewards: list[int], step: int) -> None:
    """
    Print out a sample of model completions to the console.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        reward (`list[float]`):
            List of rewards corresponding to the completions.
        step (`int`):
            Current training step number, used in the output title.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample
    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = [0.12345, 0.68789]
    >>> print_prompt_completions_sample(prompts, completions, rewards, 42)
    ╭─────────────── Step 42 ────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Reward ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩ │
    │ │ The sky is │  blue.       │   0.12 │ │
    │ ├────────────┼──────────────┼────────┤ │
    │ │ The sun is │  in the sky. │   0.68 │ │
    │ └────────────┴──────────────┴────────┘ │
    ╰────────────────────────────────────────╯
    ```
    """


    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    for prompt, completion, reward in zip(prompts, completions, rewards):
        table.add_row(Text(prompt), Text(completion), f"{reward:.2f}")  # Formatting reward to 2 decimal places
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)

# Assume necessary imports from transformers are available
# from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig, StoppingCriteriaList

class ONN_GRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        correction_instruction: str = "\n\nThere might be something wrong with the previous response. Please review it critically and provide a corrected or improved answer. If the previous response was already correct, reiterate it.",
        **kwargs,
    ):
        """
        Initializes TwoTurnGRPOTrainer.

        Args:
            *args: Positional arguments passed to the parent GRPOTrainer.
            correction_instruction (str): The instruction text inserted between
                                           the first and second turn generation.
            **kwargs: Keyword arguments passed to the parent GRPOTrainer.
        """
        super().__init__(*args, **kwargs)
        self.correction_instruction = correction_instruction
        # Tokenize the instruction once - moved inside the function as device might not be ready here
        self.instruction_ids = None
        self.instruction_mask = None


    def _prepare_instruction_ids(self, device: torch.device):
        """Helper to tokenize instruction on the correct device once."""
        if self.instruction_ids is None:
            instruction_tokens = self.processing_class(
                self.correction_instruction,
                return_tensors="pt",
                add_special_tokens=False, # Usually False for mid-sequence insertion
            )
            # Move to the correct device
            self.instruction_ids = instruction_tokens["input_ids"].squeeze(0).to(device) # Remove batch dim and move
            self.instruction_mask = instruction_tokens["attention_mask"].squeeze(0).to(device) # Remove batch dim and move
            if self.instruction_ids.shape[0] == 0:
                print("Warning: Correction instruction resulted in zero tokens.")


    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generates two turns of completions (A1 and A2) with an intermediate
        correction instruction (I), scores the final completion (A2), and
        prepares tensors for loss computation based on the full trajectory
        (Q -> A1 -> I -> A2).

        This version performs both A1 and A2 generation centrally on the main
        process before broadcasting results.
        """
        device = self.accelerator.device
        self._prepare_instruction_ids(device) # Ensure instruction is tokenized and on device

        prompts = [x["prompt"] for x in inputs] # Original prompts (Q) - List[Union[str, List[Dict]]]
        # Process prompts using chat template IF needed, get text
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # List[str]

        # --- Prepare Initial Prompt Inputs (Q) ---
        # Tokenize prompts for generation input
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # Use Trainer's _prepare_inputs explicitly for device placement
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"] # (B*G, P)

        # Apply max_prompt_length truncation if specified
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # --- Centralized Generation (A1 and A2 on Main Process) ---
        completion_ids_a1_list = [None] * len(prompts) # List of lists for A1
        completion_ids_a2_list = [None] * len(prompts) # List of lists for A2
        completions_text_a1_list = [None] * len(prompts) # Need this later for reward context

        # Gather unique prompts for efficient generation
        all_prompts_text = gather_object(prompts_text) # Gather list of strings

        if self.accelerator.is_main_process:
            # Load model weights if using vLLM and step changed
            if self.args.use_vllm and self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # --- Turn 1 Generation (Q -> A1) ---
            if self.args.use_vllm:
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                with profiling_context(self, "vLLM.generate_A1"):
                    all_outputs_a1 = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                # Unpack results correctly based on num_generations
                temp_completion_ids_a1 = []
                for outputs in all_outputs_a1:
                    for output in outputs.outputs:
                        temp_completion_ids_a1.append(output.token_ids)
                # Ensure the order matches the original duplicated all_prompts_text order
                prompt_to_completions = {prompt: [] for prompt in ordered_set_of_prompts}
                completion_idx = 0
                for outputs in all_outputs_a1:
                    prompt = outputs.prompt
                    for output in outputs.outputs:
                         prompt_to_completions[prompt].append(temp_completion_ids_a1[completion_idx])
                         completion_idx += 1

                completion_ids_a1_list = []
                for prompt in all_prompts_text:
                     # Pop completions for the current prompt to maintain order
                     completion_ids_a1_list.append(prompt_to_completions[prompt].pop(0))

            else: # Regular HF Generation for A1 (Main Process)
                print("SHOULDN'T BE HERE")
                # Requires model to be on the main process device
                temp_model = self.accelerator.unwrap_model(self.model_wrapped)
                # This assumes HF generate handles batches correctly on single device
                # Tokenize prompts ON MAIN PROCESS DEVICE for HF generate
                hf_prompt_inputs = self.processing_class(
                    all_prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                ).to(device)

                # Apply max length truncation
                hf_prompt_ids = hf_prompt_inputs["input_ids"]
                hf_prompt_mask = hf_prompt_inputs["attention_mask"]
                if self.max_prompt_length is not None:
                    hf_prompt_ids = hf_prompt_ids[:, -self.max_prompt_length :]
                    hf_prompt_mask = hf_prompt_mask[:, -self.max_prompt_length :]

                with torch.no_grad(): # Ensure no gradients during HF generate
                     prompt_completion_ids_a1 = temp_model.generate(
                         hf_prompt_ids, attention_mask=hf_prompt_mask, generation_config=self.generation_config
                     )
                prompt_length = hf_prompt_ids.size(1)
                completion_ids_a1_tensor = prompt_completion_ids_a1[:, prompt_length:] # Tensor on main device
                completion_ids_a1_list = [ids.tolist() for ids in completion_ids_a1_tensor] # Convert to list of lists


            # --- Decode A1 and Prepare Turn 2 Prompts ---
            # Decode A1 tokens (still on main process)
            # Need to handle padding/eos before decoding accurately
            temp_completion_ids_a1 = [torch.tensor(ids, device=device) for ids in completion_ids_a1_list]
            temp_completion_ids_a1_padded = pad(temp_completion_ids_a1, padding_value=self.processing_class.pad_token_id, padding_side="right")
            is_eos_a1 = temp_completion_ids_a1_padded == self.processing_class.eos_token_id
            eos_idx_a1 = torch.full((is_eos_a1.size(0),), is_eos_a1.size(1), dtype=torch.long, device=device)
            eos_idx_a1[is_eos_a1.any(dim=1)] = is_eos_a1.int().argmax(dim=1)[is_eos_a1.any(dim=1)]
            completions_text_a1_list = []
            for i in range(temp_completion_ids_a1_padded.size(0)):
                 ids_to_decode = temp_completion_ids_a1_padded[i, :eos_idx_a1[i]] # Select tokens before first EOS
                 completions_text_a1_list.append(self.processing_class.decode(ids_to_decode, skip_special_tokens=True))

            # Create Turn 2 prompts (Q + A1_text + I)
            prompts_text_turn2 = [
                q_text + a1_text + self.correction_instruction
                for q_text, a1_text in zip(all_prompts_text, completions_text_a1_list)
            ]

            # --- Turn 2 Generation ((Q + A1 + I) -> A2) ---
            if self.args.use_vllm:
                # Generate ONE completion (n=1) per prompt
                sampling_params_turn2 = deepcopy(self.sampling_params)
                sampling_params_turn2.n = 1
                with profiling_context(self, "vLLM.generate_A2"):
                     all_outputs_a2 = self.llm.generate(
                          prompts_text_turn2, sampling_params=sampling_params_turn2, use_tqdm=False
                     )
                completion_ids_a2_list = [output.outputs[0].token_ids for output in all_outputs_a2]
            else: # Regular HF Generation for A2 (Main Process)
                hf_prompt_inputs_turn2 = self.processing_class(
                    prompts_text_turn2, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                ).to(device)
                hf_prompt_ids_turn2 = hf_prompt_inputs_turn2["input_ids"]
                hf_prompt_mask_turn2 = hf_prompt_inputs_turn2["attention_mask"]

                # Ensure generation config produces only 1 sequence
                generation_config_turn2 = deepcopy(self.generation_config)
                generation_config_turn2.num_return_sequences = 1
                generation_config_turn2.num_beams = 1 # Ensure not using beam search returning multiple

                temp_model = self.accelerator.unwrap_model(self.model_wrapped) # Already unwrapped
                with torch.no_grad():
                     prompt_completion_ids_a2 = temp_model.generate(
                         hf_prompt_ids_turn2, attention_mask=hf_prompt_mask_turn2, generation_config=generation_config_turn2
                     )
                prompt_length_turn2 = hf_prompt_ids_turn2.size(1)
                completion_ids_a2_tensor = prompt_completion_ids_a2[:, prompt_length_turn2:] # Tensor on main device
                completion_ids_a2_list = [ids.tolist() for ids in completion_ids_a2_tensor] # list of lists

        # --- Broadcast Generated Token Lists to All Processes ---
        # completions_text_a1_list is only needed for reward context construction later
        # broadcast it as well if needed by non-main processes for that (though reward calc is often distributed anyway)
        completion_ids_a1_list = broadcast_object_list(completion_ids_a1_list, from_process=0)
        completion_ids_a2_list = broadcast_object_list(completion_ids_a2_list, from_process=0)
        completions_text_a1_list = broadcast_object_list(completions_text_a1_list, from_process=0) # Broadcast decoded A1 text

        # --- Process Results on Each Process ---
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        local_completion_ids_a1_list = completion_ids_a1_list[process_slice]
        local_completion_ids_a2_list = completion_ids_a2_list[process_slice]
        local_completions_text_a1_list = completions_text_a1_list[process_slice] # Needed for local reward context construction

        # Convert lists to padded tensors and create masks locally
        completion_ids_a1 = [torch.tensor(ids, device=device) for ids in local_completion_ids_a1_list]
        completion_ids_a1 = pad(completion_ids_a1, padding_value=self.processing_class.pad_token_id, padding_side="right") # (local_B*G, C1)
        is_eos_a1 = completion_ids_a1 == self.processing_class.eos_token_id
        eos_idx_a1 = torch.full((is_eos_a1.size(0),), is_eos_a1.size(1), dtype=torch.long, device=device)
        eos_idx_a1[is_eos_a1.any(dim=1)] = is_eos_a1.int().argmax(dim=1)[is_eos_a1.any(dim=1)]
        sequence_indices_a1 = torch.arange(is_eos_a1.size(1), device=device).expand(is_eos_a1.size(0), -1)
        completion_mask_a1 = (sequence_indices_a1 < eos_idx_a1.unsqueeze(1)).int() * (completion_ids_a1 != self.processing_class.pad_token_id).int() # Use < for strict before EOS

        completion_ids_a2 = [torch.tensor(ids, device=device) for ids in local_completion_ids_a2_list]
        completion_ids_a2 = pad(completion_ids_a2, padding_value=self.processing_class.pad_token_id, padding_side="right") # (local_B*G, C2)
        is_eos_a2 = completion_ids_a2 == self.processing_class.eos_token_id
        eos_idx_a2 = torch.full((is_eos_a2.size(0),), is_eos_a2.size(1), dtype=torch.long, device=device)
        eos_idx_a2[is_eos_a2.any(dim=1)] = is_eos_a2.int().argmax(dim=1)[is_eos_a2.any(dim=1)]
        sequence_indices_a2 = torch.arange(is_eos_a2.size(1), device=device).expand(is_eos_a2.size(0), -1)
        completion_mask_a2 = (sequence_indices_a2 < eos_idx_a2.unsqueeze(1)).int() * (completion_ids_a2 != self.processing_class.pad_token_id).int() # Use < for strict before EOS

        # --- Combine Sequences and Calculate Log Probabilities (Distributed) ---
        # Need prompt_ids_turn2 locally for A2 logp calculation context
        # Reconstruct Q+A1+I prompts locally using broadcasted A1 text
        local_prompts_text = prompts_text[process_slice] # Get local slice of original prompt text
        local_prompts_text_turn2 = [
            q_text + a1_text + self.correction_instruction
            for q_text, a1_text in zip(local_prompts_text, local_completions_text_a1_list)
        ]
        prompt_inputs_turn2 = self.processing_class(
            local_prompts_text_turn2, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # Use Trainer's _prepare_inputs explicitly
        prompt_inputs_turn2 = Trainer._prepare_inputs(self, prompt_inputs_turn2)
        prompt_ids_turn2, prompt_mask_turn2 = prompt_inputs_turn2["input_ids"], prompt_inputs_turn2["attention_mask"] # (local_B*G, P+C1+I_len)


        # Sequence for A1 logps: Q + A1
        input_ids_a1 = torch.cat([prompt_ids, completion_ids_a1], dim=1) # (local_B*G, P + C1)
        attention_mask_a1 = torch.cat([prompt_mask, completion_mask_a1], dim=1) # (local_B*G, P + C1)
        logits_to_keep_a1 = completion_ids_a1.size(1)

        # Sequence for A2 logps: Q + A1 + I + A2 (using locally constructed turn2 prompt)
        input_ids_a2 = torch.cat([prompt_ids_turn2, completion_ids_a2], dim=1) # (local_B*G, P+C1+I + C2)
        attention_mask_a2 = torch.cat([prompt_mask_turn2, completion_mask_a2], dim=1) # (local_B*G, P+C1+I + C2)
        logits_to_keep_a2 = completion_ids_a2.size(1)

        # --- Log Probability Calculation (Potentially Distributed via self.model/ref_model) ---
        with torch.no_grad():
            # Calculate necessary logps for A1
            if self.num_iterations > 1:
                old_per_token_logps_a1 = self._get_per_token_logps(
                    self.model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                ) # (local_B*G, C1)
            else:
                old_per_token_logps_a1 = None # Will use detached current logps later

            if self.beta == 0.0:
                ref_per_token_logps_a1 = None
            elif self.ref_model is not None:
                ref_per_token_logps_a1 = self._get_per_token_logps(
                    self.ref_model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                ) # (local_B*G, C1)
            else:
                # Ensure adapter is disabled on the *potentially wrapped* model before unwrapping
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                     # Pass the unwrapped model to _get_per_token_logps if it expects the raw model
                     # Or rely on _get_per_token_logps to handle wrapped model if it does
                     # Assuming _get_per_token_logps handles the potentially wrapped model correctly
                     ref_per_token_logps_a1 = self._get_per_token_logps(
                         self.model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                     ) # (local_B*G, C1)

            # Calculate necessary logps for A2
            if self.num_iterations > 1:
                old_per_token_logps_a2 = self._get_per_token_logps(
                    self.model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                ) # (local_B*G, C2)
            else:
                 old_per_token_logps_a2 = None # Will use detached current logps later

            if self.beta == 0.0:
                ref_per_token_logps_a2 = None
            elif self.ref_model is not None:
                ref_per_token_logps_a2 = self._get_per_token_logps(
                    self.ref_model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                ) # (local_B*G, C2)
            else:
                 with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps_a2 = self._get_per_token_logps(
                        self.model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                    ) # (local_B*G, C2)

        # --- Combine Logps for Loss Calculation Input ---
        # Pad logps based on local max lengths C1, C2 and concatenate
        max_len_c1 = completion_ids_a1.size(1)
        max_len_c2 = completion_ids_a2.size(1)

        def _combine_logps(logps1, logps2, mask1, mask2, max_c1, max_c2, device):
            if logps1 is None and logps2 is None: return None
            # Create zero tensors with target padded shape
            batch_size = mask1.size(0)
            padded_logps1 = torch.zeros((batch_size, max_c1), dtype=torch.float, device=device)
            if logps1 is not None:
                 current_c1 = logps1.size(1)
                 padded_logps1[:, :current_c1] = logps1 * mask1[:,:current_c1] # Apply mask
            padded_logps2 = torch.zeros((batch_size, max_c2), dtype=torch.float, device=device)
            if logps2 is not None:
                 current_c2 = logps2.size(1)
                 padded_logps2[:, :current_c2] = logps2 * mask2[:,:current_c2] # Apply mask
            return torch.cat((padded_logps1, padded_logps2), dim=1) # (local_B*G, max_C1 + max_C2)

        old_per_token_logps = _combine_logps(old_per_token_logps_a1, old_per_token_logps_a2, completion_mask_a1, completion_mask_a2, max_len_c1, max_len_c2, device)
        ref_per_token_logps = _combine_logps(ref_per_token_logps_a1, ref_per_token_logps_a2, completion_mask_a1, completion_mask_a2, max_len_c1, max_len_c2, device)

        # --- Decode Final Completions (A2) and Calculate Rewards (Distributed) ---
        # Decode A2 locally for reward calculation and logging
        local_completions_text_a2_list = []
        for ids, mask in zip(completion_ids_a2, completion_mask_a2):
             # Decode tokens selected by the mask (handles padding and EOS)
             masked_ids = ids[mask.bool()]
             local_completions_text_a2_list.append(self.processing_class.decode(masked_ids, skip_special_tokens=True))

        # Prepare inputs for reward functions based on A2 using local data
        reward_prompts = local_prompts_text_turn2 # Q+A1+I context
        reward_completions = local_completions_text_a2_list # A2

        # Prepare reward model inputs (handle conversational vs text)
        # Using the same logic as before, but operating on local data slices
        if is_conversational(inputs[0]):
            # This requires the original `prompts` (List[List[Dict]]) sliced locally
            local_prompts_structured = prompts[process_slice]
            full_conversations = []
            for q_list, a1_text, a2_text in zip(local_prompts_structured, local_completions_text_a1_list, local_completions_text_a2_list):
                conv = deepcopy(q_list)
                conv.append({"role": "assistant", "content": a1_text})
                # Decide if/how to represent I for the reward model
                conv.append({"role": "assistant", "content": a2_text})
                full_conversations.append(conv)

            reward_inputs_texts = []
            # Assuming a single reward function/tokenizer for simplicity here
            # Needs adjustment if multiple reward funcs have different tokenizers
            reward_processing_class = self.reward_processing_classes[0]
            for conv in full_conversations:
                 processed = apply_chat_template({"messages": conv}, reward_processing_class)
                 reward_inputs_texts.append(processed["text"])
        else:
            reward_inputs_texts = [p + c for p, c in zip(reward_prompts, reward_completions)]

        # Calculate rewards (potentially distributed via reward models)
        rewards_per_func = torch.zeros(len(local_prompts_text), len(self.reward_funcs), device=device) # Shape (local_B*G, num_rewards)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Use appropriate reward_inputs_texts based on conversational/text and function index 'i' if tokenizers differ
            # Simplified here to use the pre-calculated reward_inputs_texts
            current_reward_inputs_texts = reward_inputs_texts # Needs adjustment if reward funcs need different formats

            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                with profiling_context(self, reward_func_name):
                     reward_inputs = reward_processing_class(
                         current_reward_inputs_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                     )
                     # Use Trainer's _prepare_inputs explicitly
                     reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                     with torch.inference_mode():
                         rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else: # Function-based reward
                reward_func_name = reward_func.__name__
                with profiling_context(self, reward_func_name):
                    # Function needs local data slice
                    local_inputs_slice = inputs[process_slice] # Get the slice of original dict inputs
                    keys = [key for key in local_inputs_slice[0] if key not in ["prompt", "completion"]]
                    # Create kwargs from the *local slice* of inputs
                    reward_kwargs = {key: [example[key] for example in local_inputs_slice] for key in keys}
                    output_reward_func = reward_func(
                        prompts=reward_prompts, 
                        completions=reward_completions, 
                        **reward_kwargs
                    )
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        # --- Calculate Advantages (Requires Gathering Rewards) and Log Metrics ---
        # Gather rewards from all processes
        # Shape of rewards_per_func is (local_B*G, num_rewards) -> gather -> (B*G, num_rewards)
        gathered_rewards_per_func = gather(rewards_per_func)

        # Compute combined reward score (on gathered tensor)
        gathered_rewards = (gathered_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1) # (B*G,)

        # Compute grouped-wise rewards (original GRPO logic) using gathered rewards
        num_total_samples = gathered_rewards.size(0)
        num_prompts_total = num_total_samples // self.num_generations
        mean_grouped_rewards = gathered_rewards.view(num_prompts_total, self.num_generations).mean(dim=1)
        std_grouped_rewards = gathered_rewards.view(num_prompts_total, self.num_generations).std(dim=1)

        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B*G,)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B*G,)
        gathered_advantages = gathered_rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            gathered_advantages = gathered_advantages / (std_grouped_rewards + 1e-4) # (B*G,)

        # Get the advantages corresponding to the local process slice
        advantages = gathered_advantages[process_slice] # (local_B*G,)

        # --- Logging ---
        mode = "eval" if self.control.should_evaluate else "train"

        # Combined completion IDs and mask for length logging
        completion_ids = torch.cat((completion_ids_a1, completion_ids_a2), dim=1) # (local_B*G, C1+C2)
        completion_mask = torch.cat((completion_mask_a1, completion_mask_a2), dim=1) # (local_B*G, C1+C2)

        # Gather lengths for accurate mean calculation
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length) # Logs total length A1+A2

        # Log mean reward per function across all processes
        reward_per_func_mean = gathered_rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            # Name extraction logic from original code
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        # Log overall mean reward and mean std dev across groups
        self._metrics[mode]["reward"].append(gathered_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(gathered_rewards.view(num_prompts_total, self.num_generations).std(dim=1).mean().item()) # Mean of std devs

        # Log completions (Gather all necessary text)
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
             # We already gathered all_prompts_text
             # Gather A1 text (already broadcasted, just need gather for main process)
             all_completions_text_a1 = gather_object(local_completions_text_a1_list)
             # Gather A2 text
             all_completions_text_a2 = gather_object(local_completions_text_a2_list)
             # Use gathered rewards
             all_rewards_to_log = gathered_rewards.tolist()

             if self.accelerator.is_main_process:
                 print(f"\n--- Step {self.state.global_step} Two-Turn Completions ---")
                 num_to_log = min(1, len(all_prompts_text))
                 for i in range(num_to_log): # Log first few examples
                      print(f"\n[Example {i+1}]")
                      print(f"  Prompt (Q): {all_prompts_text[i]}")
                      print(f"  Turn 1 (A1): {all_completions_text_a1[i]}")
                      print(f"  Turn 2 (A2): {all_completions_text_a2[i]}")
                      print(f"  Reward: {all_rewards_to_log[i]:.4f}")
                 print("------------------------------------\n")

                 if self.args.report_to and "wandb" in self.args.report_to and wandb and wandb.run:
                     import pandas as pd
                     table = {
                         "step": [str(self.state.global_step)] * len(all_rewards_to_log),
                         "prompt": all_prompts_text,
                         "completion_A1": all_completions_text_a1,
                         "completion_A2": all_completions_text_a2,
                         "reward": all_rewards_to_log,
                     }
                     df = pd.DataFrame(table)
                     wandb.log({"two_turn_completions": wandb.Table(dataframe=df)})


        # --- Return results for loss computation ---
        # Return tensors corresponding to the *local process slice*
        return {
            "prompt_ids": prompt_ids, # (local_B*G, P)
            "prompt_mask": prompt_mask, # (local_B*G, P)
            "completion_ids": completion_ids, # (local_B*G, C1+C2) - Combined A1+A2 IDs
            "completion_mask": completion_mask, # (local_B*G, C1+C2) - Combined A1+A2 mask
            "old_per_token_logps": old_per_token_logps, # (local_B*G, C1+C2) - Combined A1+A2 logps
            "ref_per_token_logps": ref_per_token_logps, # (local_B*G, C1+C2) - Combined A1+A2 ref logps
            "advantages": advantages, # (local_B*G,) - Single advantage value per trajectory
        }