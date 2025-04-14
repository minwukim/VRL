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


# Helper function provided by user (can be defined outside the class)
# Note: Its utility depends heavily on the self.a1_prompt_format
def extract_a1_text_from_final_prompt(prompt_string, a1_prompt_format, instruction):
    """
    Attempts to extract A1 text given the final prompt string and the format used.
    This is non-trivial and fragile.
    """
    try:
        # Simple approach: assume instruction marks the end of A1
        # Find the start of the instruction within the formatted string
        instruction_start_index = prompt_string.rfind(instruction)
        if instruction_start_index == -1:
            return None # Instruction not found

        # Find the end of the original prompt (Q) - this is the hard part
        # We might need to rely on fixed markers if the prompt format guarantees them
        # Example: If A1 always starts after "<|im_start|>assistant\n" from Q.
        q_end_marker = "<|im_start|>assistant\n" # Or similar based on Q format
        q_end_index = prompt_string.find(q_end_marker)
        if q_end_index == -1:
            a1_start_index = 0 # Assume Q was just text without markers
        else:
            a1_start_index = q_end_index + len(q_end_marker)

        # Find the start of the user turn that contains the instruction
        user_turn_marker = "<|im_start|>user" # Assuming this precedes instruction
        user_turn_index = prompt_string.rfind(user_turn_marker, 0, instruction_start_index)

        if user_turn_index != -1:
            # A1 is likely between end of Q marker and start of user turn marker
            a1_end_marker = "<|im_end|>" # Usually ends assistant turn
            a1_end_index = prompt_string.rfind(a1_end_marker, a1_start_index, user_turn_index)
            if a1_end_index != -1:
                 return prompt_string[a1_start_index:a1_end_index].strip()

        # Fallback: Take text between assumed end of Q and start of instruction
        return prompt_string[a1_start_index:instruction_start_index].strip()

    except Exception as e:
        print(f"Warning: Error extracting A1 text: {e}")
        return None


class ONN_GRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer variant for two-turn generation (Q -> n * A1 -> n * A2).

    1. Generates 'num_generations' (n) initial responses (A1) for each unique prompt (Q).
    2. For each (Q, A1) pair, constructs a new prompt including a correction instruction (I).
    3. Generates one refined response (A2) for each (Q, A1, I) prompt.
    4. Calculates reward based on the final A2.
    5. Calculates loss based on generated tokens in *both* A1 and A2, using the advantage from A2's reward.
    """
    def __init__(
        self,
        *args,
        correction_instruction: str = "\nUser: There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Maintain the format of: <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>.\nAssistant: <think>",
        # correction_instruction: str = "There might be something wrong with the previous response. Please review it critically and provide a corrected or improved answer. If the previous response was already correct, reiterate it.",
        a1_prompt_format: str = "{prompt}{completion}{instruction}",
        # a1_prompt_format: str = "{prompt}{completion}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        
        **kwargs,
    ):
        """
        Initializes TwoTurnGRPOTrainer_1nn.

        Args:
            *args: Positional arguments passed to the parent GRPOTrainer.
            correction_instruction (str): The instruction text inserted between turns.
            a1_prompt_format (str): Format string for the A2 prompt. Must contain
                                    {prompt} (original Q text/template),
                                    {completion} (A1 text), and {instruction}.
            **kwargs: Keyword arguments passed to the parent GRPOTrainer.
        """
        super().__init__(*args, **kwargs)
        self.correction_instruction = correction_instruction
        self.a1_prompt_format = a1_prompt_format
        # Validate format string immediately
        if "{prompt}" not in self.a1_prompt_format or \
           "{completion}" not in self.a1_prompt_format or \
           "{instruction}" not in self.a1_prompt_format:
               raise ValueError("a1_prompt_format must contain {prompt}, {completion}, and {instruction}")


    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        print("here1: Entering _generate_and_score_completions")
        device = self.accelerator.device

        # --- Prepare Initial Prompt Inputs (Q) ---
        # Keep original structure (e.g., for chat) if needed by maybe_apply_chat_template
        prompts_structured = [x["prompt"] for x in inputs]
        # Get flat list of text prompts, applying template if necessary
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # List[str], length B*G
        num_local_prompts = len(prompts_text)

        print(f"here2: Processed initial prompts. Local batch size: {num_local_prompts}")

        # Tokenize local Q prompts for later use in A1 logp calculation context
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # Explicit call to base Trainer
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"] # (local_B*G, P)

        # Apply max_prompt_length truncation if specified
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # --- Centralized Generation (A1 and A2 on Main Process) ---
        # Gather all initial prompt texts to main process
        all_prompts_text = gather_object(prompts_text) # List[str], length B_total * G
        num_total_prompts = len(all_prompts_text)

        # Initialize placeholders correctly *before* the main process check
        completion_ids_a1_list = [None] * num_total_prompts
        completion_ids_a2_list = [None] * num_total_prompts
        completions_text_a1_list = [None] * num_total_prompts
        prompts_text_turn2_all = [None] * num_total_prompts # Store generated A2 prompts for logging/reward

        # --- Main Process Generation ---
        if self.accelerator.is_main_process:
            print(f"here3: Main process starting generation. Total prompts: {num_total_prompts}")
            # Check if vLLM is enabled (should be done *after* checking if main process)
            if self.args.use_vllm:

                # --- Turn 1 Generation (Q -> n * A1) ---
                ordered_unique_prompts = list(dict.fromkeys(all_prompts_text)) # Preserves order
                num_unique_prompts = len(ordered_unique_prompts)
                print(f"here4: Generating A1 for {num_unique_prompts} unique prompts, n={self.num_generations}")

                # Use original sampling params (contains n=num_generations)
                try:
                    with profiling_context(self, "vLLM.generate_A1"):
                        all_outputs_a1 = self.llm.generate(
                            ordered_unique_prompts, sampling_params=self.sampling_params, use_tqdm=False
                        )
                except Exception as e:
                    print(f"ERROR during vLLM A1 generation: {e}")
                    # Handle error: maybe re-raise or attempt fallback/dummy data
                    raise e # Re-raise for now

                # Process results carefully, ensuring alignment and handling generation count issues
                completion_ids_a1_nested = [] # List[ List[ids] ]
                completions_text_a1_nested = [] # List[ List[str] ]
                prompt_outputs_map = {req_output.prompt: req_output.outputs for req_output in all_outputs_a1}

                for unique_prompt in ordered_unique_prompts:
                    ids_for_prompt = []
                    texts_for_prompt = []
                    if unique_prompt not in prompt_outputs_map:
                         print(f"Warning: Unique prompt not found in vLLM output map! Prompt: {unique_prompt[:100]}...")
                         # Handle missing output: pad with empty lists?
                         outputs_for_prompt = []
                    else:
                         outputs_for_prompt = prompt_outputs_map[unique_prompt]

                    # Verify count and handle discrepancies
                    if len(outputs_for_prompt) != self.num_generations:
                        print(f"Warning: Expected {self.num_generations} A1 outputs for prompt, but got {len(outputs_for_prompt)}. Prompt: {unique_prompt[:100]}...")
                        # Strategy: Use available, pad with last valid one or empty if none
                        outputs_to_use = list(outputs_for_prompt) # Copy
                        if not outputs_to_use: # No outputs at all? Pad with empty.
                            pad_output_ids = []
                            pad_output_text = ""
                            print(f"  -> Padding with empty A1 for prompt: {unique_prompt[:100]}...")
                        else:
                            pad_output_ids = outputs_to_use[-1].token_ids
                            pad_output_text = self.processing_class.decode(pad_output_ids, skip_special_tokens=True)

                        while len(outputs_to_use) < self.num_generations:
                            outputs_to_use.append(None) # Placeholder for loop, will use pad_output below

                        temp_ids_for_prompt = []
                        temp_texts_for_prompt = []
                        for i in range(self.num_generations):
                            if i < len(outputs_for_prompt):
                                temp_ids_for_prompt.append(outputs_for_prompt[i].token_ids)
                                temp_texts_for_prompt.append(self.processing_class.decode(outputs_for_prompt[i].token_ids, skip_special_tokens=True))
                            else:
                                temp_ids_for_prompt.append(deepcopy(pad_output_ids)) # Use deepcopy for lists
                                temp_texts_for_prompt.append(pad_output_text)
                        ids_for_prompt = temp_ids_for_prompt
                        texts_for_prompt = temp_texts_for_prompt
                    else: # Correct number generated
                        for output in outputs_for_prompt:
                            ids_for_prompt.append(output.token_ids)
                            texts_for_prompt.append(self.processing_class.decode(output.token_ids, skip_special_tokens=True))

                    completion_ids_a1_nested.append(ids_for_prompt)
                    completions_text_a1_nested.append(texts_for_prompt)


                print("here5: Processing A1 results, preparing A2 prompts.")
                # Flatten nested lists to match the order of `all_prompts_text`
                # Re-initialize flat lists before populating
                completion_ids_a1_list = []
                completions_text_a1_list = []
                unique_prompt_to_idx = {prompt: i for i, prompt in enumerate(ordered_unique_prompts)}
                gen_idx_counters = {prompt: 0 for prompt in ordered_unique_prompts} # Track usage per prompt

                for i, prompt_text in enumerate(all_prompts_text):
                    if prompt_text not in unique_prompt_to_idx:
                         print(f"ERROR: Original prompt text not found in unique map during flattening! Index {i}, Prompt: {prompt_text[:100]}")
                         # Handle error: append empty/None or raise? Append None for now.
                         completion_ids_a1_list.append(None)
                         completions_text_a1_list.append(None)
                         continue

                    unique_idx = unique_prompt_to_idx[prompt_text]
                    gen_idx = gen_idx_counters[prompt_text]

                    if unique_idx >= len(completion_ids_a1_nested) or gen_idx >= len(completion_ids_a1_nested[unique_idx]):
                         print(f"ERROR: Index out of bounds during flattening! unique_idx={unique_idx}, gen_idx={gen_idx}, prompt={prompt_text[:100]}")
                         completion_ids_a1_list.append(None)
                         completions_text_a1_list.append(None)
                         # Don't increment counter if we failed
                    else:
                         completion_ids_a1_list.append(completion_ids_a1_nested[unique_idx][gen_idx])
                         completions_text_a1_list.append(completions_text_a1_nested[unique_idx][gen_idx])
                         gen_idx_counters[prompt_text] += 1 # Increment counter only on success


                # --- Prepare Turn 2 Prompts (Q + A1 + I) ---
                # Re-initialize list for A2 prompts
                prompts_text_turn2 = []
                for q_text, a1_text in zip(all_prompts_text, completions_text_a1_list):
                    if a1_text is None: # Handle potential None from error above
                         print(f"Warning: Skipping A2 prompt generation for prompt due to missing A1. Q: {q_text[:100]}...")
                         prompts_text_turn2.append(None) # Mark A2 prompt as invalid
                         continue

                    try:
                        turn2_prompt = self.a1_prompt_format.format(
                            prompt=q_text,
                            completion=a1_text,
                            instruction=self.correction_instruction
                        )
                        prompts_text_turn2.append(turn2_prompt)
                    except Exception as e:
                         print(f"ERROR formatting A2 prompt: {e}. Q: {q_text[:100]}, A1: {a1_text[:100]}")
                         prompts_text_turn2.append(None)

                prompts_text_turn2_all = prompts_text_turn2 # Store for broadcasting

                print(f"here6: Prepared {len(prompts_text_turn2)} prompts for A2 generation.")

                # Filter out None prompts before sending to vLLM
                valid_prompts_turn2 = [p for p in prompts_text_turn2 if p is not None]
                valid_indices_turn2 = [i for i, p in enumerate(prompts_text_turn2) if p is not None]

                if not valid_prompts_turn2:
                    print("Warning: No valid prompts remaining for A2 generation.")
                    # Populate final A2 list with None or empty lists
                    completion_ids_a2_list = [[] for _ in range(num_total_prompts)] # Use empty list as default
                else:
                    # --- Turn 2 Generation ((Q + A1 + I) -> 1 * A2) ---
                    sampling_params_turn2 = deepcopy(self.sampling_params)
                    sampling_params_turn2.n = 1
                    try:
                        with profiling_context(self, "vLLM.generate_A2"):
                             all_outputs_a2 = self.llm.generate(
                                  valid_prompts_turn2, # Only send valid prompts
                                  sampling_params=sampling_params_turn2,
                                  use_tqdm=False
                             )
                    except Exception as e:
                        print(f"ERROR during vLLM A2 generation: {e}")
                        raise e

                    # Map results back to the original N slots, filling Nones for failed prompts
                    temp_completion_ids_a2 = [output.outputs[0].token_ids for output in all_outputs_a2]
                    # Re-initialize a2 list with default (e.g., empty list)
                    completion_ids_a2_list = [[] for _ in range(num_total_prompts)]
                    for i, result_idx in enumerate(valid_indices_turn2):
                         if i < len(temp_completion_ids_a2):
                              completion_ids_a2_list[result_idx] = temp_completion_ids_a2[i]
                         else:
                              print(f"Warning: Mismatch between valid A2 prompts ({len(valid_indices_turn2)}) and A2 outputs ({len(temp_completion_ids_a2)}). Index {i}")
                              # Keep the default empty list for completion_ids_a2_list[result_idx]

                print(f"here7: Finished generating A2. Result list length: {len(completion_ids_a2_list)}")
                # Final check for safety before broadcasting
                if len(completion_ids_a1_list) != num_total_prompts:
                     print(f"FATAL: A1 list length mismatch before broadcast! Got {len(completion_ids_a1_list)}, expected {num_total_prompts}")
                     # Attempt to fix or raise error
                     completion_ids_a1_list = (completion_ids_a1_list + [None] * num_total_prompts)[:num_total_prompts]

                if len(completion_ids_a2_list) != num_total_prompts:
                     print(f"FATAL: A2 list length mismatch before broadcast! Got {len(completion_ids_a2_list)}, expected {num_total_prompts}")
                     completion_ids_a2_list = (completion_ids_a2_list + [[]] * num_total_prompts)[:num_total_prompts] # Pad with empty lists

                if len(completions_text_a1_list) != num_total_prompts:
                     print(f"FATAL: A1 text list length mismatch before broadcast! Got {len(completions_text_a1_list)}, expected {num_total_prompts}")
                     completions_text_a1_list = (completions_text_a1_list + [None] * num_total_prompts)[:num_total_prompts]

                if len(prompts_text_turn2_all) != num_total_prompts:
                     print(f"FATAL: A2 prompt text list length mismatch before broadcast! Got {len(prompts_text_turn2_all)}, expected {num_total_prompts}")
                     prompts_text_turn2_all = (prompts_text_turn2_all + [None] * num_total_prompts)[:num_total_prompts]


            else: # Non-vLLM (Not supported for this logic currently)
                raise NotImplementedError("Two-turn generation with 1:n:n logic currently requires vLLM.")

        else: # Non-main process
            print(f"here3: Non-main process {self.accelerator.process_index} waiting for broadcast. Placeholders initialized.")
            # Placeholders already initialized with correct size and None values

        # --- Broadcast Generated Lists from Main Process ---
        print(f"here8: Broadcasting results from main process. Process {self.accelerator.process_index}")
        # Ensure data being broadcast is not None itself if possible errors occurred on main
        if self.accelerator.is_main_process:
            # Replace potential None lists with lists of empty lists/Nones if needed before broadcast
            completion_ids_a1_list = completion_ids_a1_list if completion_ids_a1_list is not None else [[] for _ in range(num_total_prompts)]
            completion_ids_a2_list = completion_ids_a2_list if completion_ids_a2_list is not None else [[] for _ in range(num_total_prompts)]
            completions_text_a1_list = completions_text_a1_list if completions_text_a1_list is not None else [None for _ in range(num_total_prompts)]
            prompts_text_turn2_all = prompts_text_turn2_all if prompts_text_turn2_all is not None else [None for _ in range(num_total_prompts)]


        completion_ids_a1_list = broadcast_object_list(completion_ids_a1_list, from_process=0)
        completion_ids_a2_list = broadcast_object_list(completion_ids_a2_list, from_process=0)
        completions_text_a1_list = broadcast_object_list(completions_text_a1_list, from_process=0) # Broadcast decoded A1 text
        prompts_text_turn2_all = broadcast_object_list(prompts_text_turn2_all, from_process=0) # Broadcast A2 prompts

        # --- Process Results Locally on Each Process ---
        print(f"here9: Processing broadcasted results locally. Process {self.accelerator.process_index}")
        process_slice = slice(
            self.accelerator.process_index * num_local_prompts,
            (self.accelerator.process_index + 1) * num_local_prompts,
        )
        local_completion_ids_a1_list = completion_ids_a1_list[process_slice]
        local_completion_ids_a2_list = completion_ids_a2_list[process_slice]
        local_completions_text_a1_list = completions_text_a1_list[process_slice]
        local_prompts_text_turn2 = prompts_text_turn2_all[process_slice] # Get local slice of A2 prompts

        # Handle potential None values received from broadcast (if errors occurred on main)
        # Replace None with empty lists [] for token IDs before converting to tensor
        safe_local_completion_ids_a1 = [ids if ids is not None else [] for ids in local_completion_ids_a1_list]
        safe_local_completion_ids_a2 = [ids if ids is not None else [] for ids in local_completion_ids_a2_list]

        # Convert lists to padded tensors and create masks locally
        try:
            completion_ids_a1 = [torch.tensor(ids, device=device, dtype=torch.long) for ids in safe_local_completion_ids_a1]
            completion_ids_a1 = pad(completion_ids_a1, padding_value=self.processing_class.pad_token_id, padding_side="right") # (local_B*G, C1)
        except Exception as e:
            print(f"ERROR padding A1 IDs on process {self.accelerator.process_index}: {e}")
            print("A1 IDs received:", safe_local_completion_ids_a1)
            raise e # Stop execution

        try:
            completion_ids_a2 = [torch.tensor(ids, device=device, dtype=torch.long) for ids in safe_local_completion_ids_a2]
            completion_ids_a2 = pad(completion_ids_a2, padding_value=self.processing_class.pad_token_id, padding_side="right") # (local_B*G, C2)
        except Exception as e:
            print(f"ERROR padding A2 IDs on process {self.accelerator.process_index}: {e}")
            print("A2 IDs received:", safe_local_completion_ids_a2)
            raise e

        # --- Create Masks ---
        is_eos_a1 = completion_ids_a1 == self.processing_class.eos_token_id
        eos_idx_a1 = torch.full((is_eos_a1.size(0),), is_eos_a1.size(1), dtype=torch.long, device=device)
        has_eos_a1 = is_eos_a1.any(dim=1)
        if has_eos_a1.any(): eos_idx_a1[has_eos_a1] = is_eos_a1.int().argmax(dim=1)[has_eos_a1]
        sequence_indices_a1 = torch.arange(is_eos_a1.size(1), device=device).expand(is_eos_a1.size(0), -1)
        completion_mask_a1 = (sequence_indices_a1 <= eos_idx_a1.unsqueeze(1)).int() * (completion_ids_a1 != self.processing_class.pad_token_id).int()

        is_eos_a2 = completion_ids_a2 == self.processing_class.eos_token_id
        eos_idx_a2 = torch.full((is_eos_a2.size(0),), is_eos_a2.size(1), dtype=torch.long, device=device)
        has_eos_a2 = is_eos_a2.any(dim=1)
        if has_eos_a2.any(): eos_idx_a2[has_eos_a2] = is_eos_a2.int().argmax(dim=1)[has_eos_a2]
        sequence_indices_a2 = torch.arange(is_eos_a2.size(1), device=device).expand(is_eos_a2.size(0), -1)
        completion_mask_a2 = (sequence_indices_a2 <= eos_idx_a2.unsqueeze(1)).int() * (completion_ids_a2 != self.processing_class.pad_token_id).int()

        # --- Prepare Context for A2 Log Probability Calculation ---
        print(f"here10: Preparing context for A2 logp calculation. Process {self.accelerator.process_index}")
        # Tokenize the received A2 prompts locally
        # Handle potential None prompts received
        safe_local_prompts_turn2 = [p if p is not None else "" for p in local_prompts_text_turn2] # Replace None with empty string
        prompt_inputs_turn2 = self.processing_class(
            safe_local_prompts_turn2, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs_turn2 = Trainer._prepare_inputs(self, prompt_inputs_turn2)
        prompt_ids_turn2, prompt_mask_turn2 = prompt_inputs_turn2["input_ids"], prompt_inputs_turn2["attention_mask"] # (local_B*G, P+C1+I_len)


        # --- Log Probability Calculation (Potentially Distributed via self.model/ref_model) ---
        print(f"here11: Calculating log probabilities. Process {self.accelerator.process_index}")
        # Sequence for A1 logps: Q + A1
        input_ids_a1 = torch.cat([prompt_ids, completion_ids_a1], dim=1) # (local_B*G, P + C1)
        attention_mask_a1 = torch.cat([prompt_mask, completion_mask_a1], dim=1) # (local_B*G, P + C1)
        logits_to_keep_a1 = completion_ids_a1.size(1)

        # Sequence for A2 logps: (Q + A1 + I) + A2
        input_ids_a2 = torch.cat([prompt_ids_turn2, completion_ids_a2], dim=1) # (local_B*G, P+C1+I_len + C2)
        attention_mask_a2 = torch.cat([prompt_mask_turn2, completion_mask_a2], dim=1) # (local_B*G, P+C1+I_len + C2)
        logits_to_keep_a2 = completion_ids_a2.size(1)

        old_per_token_logps_a1, ref_per_token_logps_a1 = None, None
        old_per_token_logps_a2, ref_per_token_logps_a2 = None, None

        # Calculate logps only if completions have non-zero length
        calc_logps_a1 = logits_to_keep_a1 > 0
        calc_logps_a2 = logits_to_keep_a2 > 0

        with torch.no_grad():
            # A1 Logps
            if calc_logps_a1:
                if self.num_iterations > 1:
                    old_per_token_logps_a1 = self._get_per_token_logps(
                        self.model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                    ) # (local_B*G, C1)
                if self.beta > 0.0:
                    if self.ref_model is not None:
                        ref_per_token_logps_a1 = self._get_per_token_logps(
                            self.ref_model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                        ) # (local_B*G, C1)
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                             ref_per_token_logps_a1 = self._get_per_token_logps(
                                 self.model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                             ) # (local_B*G, C1)

            # A2 Logps
            if calc_logps_a2:
                if self.num_iterations > 1:
                    old_per_token_logps_a2 = self._get_per_token_logps(
                        self.model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                    ) # (local_B*G, C2)
                if self.beta > 0.0:
                    if self.ref_model is not None:
                        ref_per_token_logps_a2 = self._get_per_token_logps(
                            self.ref_model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                        ) # (local_B*G, C2)
                    else:
                         with self.accelerator.unwrap_model(self.model).disable_adapter():
                            ref_per_token_logps_a2 = self._get_per_token_logps(
                                self.model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                            ) # (local_B*G, C2)

        # --- Combine Logps, Completions, and Masks for Loss Calculation Input ---
        max_len_c1 = completion_ids_a1.size(1)
        max_len_c2 = completion_ids_a2.size(1)

        def _combine_logps(logps1, logps2, mask1, mask2, max_c1, max_c2, device):
            # Handle cases where one or both logps might be None (e.g., if generation failed)
            batch_size = mask1.size(0)
            # Pad logps1
            if logps1 is not None:
                padded_logps1 = torch.zeros((batch_size, max_c1), dtype=torch.float, device=device)
                current_c1 = logps1.size(1)
                padded_logps1[:, :current_c1] = logps1 * mask1[:,:current_c1] # Apply mask
            else: # If logps1 is None, create zero tensor
                padded_logps1 = torch.zeros((batch_size, max_c1), dtype=torch.float, device=device)

            # Pad logps2
            if logps2 is not None:
                padded_logps2 = torch.zeros((batch_size, max_c2), dtype=torch.float, device=device)
                current_c2 = logps2.size(1)
                padded_logps2[:, :current_c2] = logps2 * mask2[:,:current_c2] # Apply mask
            else: # If logps2 is None, create zero tensor
                padded_logps2 = torch.zeros((batch_size, max_c2), dtype=torch.float, device=device)

            return torch.cat((padded_logps1, padded_logps2), dim=1) # (local_B*G, max_C1 + max_C2)

        old_per_token_logps = _combine_logps(old_per_token_logps_a1, old_per_token_logps_a2, completion_mask_a1, completion_mask_a2, max_len_c1, max_len_c2, device)
        ref_per_token_logps = _combine_logps(ref_per_token_logps_a1, ref_per_token_logps_a2, completion_mask_a1, completion_mask_a2, max_len_c1, max_len_c2, device)

        # Combine completion IDs and Masks
        completion_ids = torch.cat((completion_ids_a1, completion_ids_a2), dim=1) # (local_B*G, C1+C2)
        completion_mask = torch.cat((completion_mask_a1, completion_mask_a2), dim=1) # (local_B*G, C1+C2)

        # --- Decode Final Completions (A2) and Calculate Rewards (Distributed) ---
        print(f"here12: Calculating Rewards. Process {self.accelerator.process_index}")
        # Decode A2 locally for reward calculation and logging
        local_completions_text_a2_list = []
        for ids, mask in zip(completion_ids_a2, completion_mask_a2):
             masked_ids = ids[mask.bool()] # Use boolean indexing with the mask
             # Handle empty masked_ids if A2 generation failed completely
             if masked_ids.numel() == 0:
                 local_completions_text_a2_list.append("")
             else:
                 local_completions_text_a2_list.append(self.processing_class.decode(masked_ids, skip_special_tokens=True))

        # Prepare inputs for reward functions based on A2 using local data
        reward_prompts = local_prompts_text_turn2 # Q+A1+I context (use the safe version)
        reward_completions = local_completions_text_a2_list # A2

        # Prepare reward model inputs (handle conversational vs text)
        # Use the same logic as before, but operating on local data slices
        # Check for None in local_prompts_text_turn2 before processing
        valid_reward_indices = [i for i, p in enumerate(reward_prompts) if p is not None and reward_completions[i] is not None]

        if not valid_reward_indices:
             print(f"Warning: No valid prompt/completion pairs for reward calculation on process {self.accelerator.process_index}")
             # Need to handle this case to avoid errors, e.g., return zero rewards
             rewards_per_func = torch.zeros(num_local_prompts, len(self.reward_funcs), device=device)
        else:
             # Filter inputs based on valid indices
             filtered_reward_prompts = [reward_prompts[i] for i in valid_reward_indices]
             filtered_reward_completions = [reward_completions[i] for i in valid_reward_indices]
             filtered_inputs = [inputs[i] for i in valid_reward_indices] # Filter original inputs dict list
             filtered_prompts_structured = [prompts_structured[i] for i in valid_reward_indices] # Filter structured Q prompts
             filtered_completions_text_a1 = [local_completions_text_a1_list[i] for i in valid_reward_indices] # Filter A1 text

             if is_conversational(filtered_inputs[0]): # Check filtered input
                 full_conversations = []
                 for q_list, a1_text, a2_text in zip(filtered_prompts_structured, filtered_completions_text_a1, filtered_reward_completions):
                     conv = deepcopy(q_list)
                     conv.append({"role": "assistant", "content": a1_text})
                     conv.append({"role": "assistant", "content": a2_text}) # Simplified representation
                     full_conversations.append(conv)

                 reward_inputs_texts = []
                 # Assume a single reward function/tokenizer for simplicity here
                 if self.reward_processing_classes:
                     reward_processing_class = self.reward_processing_classes[0]
                     for conv in full_conversations:
                          try:
                               processed = apply_chat_template({"messages": conv}, reward_processing_class)
                               reward_inputs_texts.append(processed["text"])
                          except Exception as e:
                               print(f"Error applying chat template for reward: {e}")
                               reward_inputs_texts.append("") # Append empty on error
                 else:
                     print("Warning: reward_processing_classes is empty.")
                     reward_inputs_texts = [""] * len(full_conversations)

             else: # Text format
                 reward_inputs_texts = [p + c for p, c in zip(filtered_reward_prompts, filtered_reward_completions)]

             # Calculate rewards for the valid items
             temp_rewards_per_func = torch.zeros(len(valid_reward_indices), len(self.reward_funcs), device=device)
             for i, (reward_func, reward_processing_class) in enumerate(
                 zip(self.reward_funcs, self.reward_processing_classes)
             ):
                 # Use appropriate reward_inputs_texts
                 current_reward_inputs_texts = reward_inputs_texts

                 if isinstance(reward_func, nn.Module):
                     reward_func_name = f"reward_{i}" # Simpler name
                     with profiling_context(self, reward_func_name):
                          reward_inputs = reward_processing_class(
                              current_reward_inputs_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                          )
                          reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                          try:
                              with torch.inference_mode():
                                  # Check if reward_func returns logits or scores directly
                                  output = reward_func(**reward_inputs)
                                  if hasattr(output, 'logits'):
                                      temp_rewards_per_func[:, i] = output.logits[:, 0]
                                  elif hasattr(output, 'scores'): # Adapt if needed
                                      temp_rewards_per_func[:, i] = output.scores
                                  else: # Assume direct output is score tensor
                                      temp_rewards_per_func[:, i] = output[:,0] if output.ndim > 1 else output

                          except Exception as e:
                              print(f"Error during reward model {i} inference: {e}")
                              # temp_rewards_per_func[:, i] remains zero

                 else: # Function-based reward
                     reward_func_name = reward_func.__name__
                     with profiling_context(self, reward_func_name):
                         # Pass only valid data to the reward function
                         keys = [key for key in filtered_inputs[0] if key not in ["prompt", "completion"]]
                         reward_kwargs = {key: [example[key] for example in filtered_inputs] for key in keys}
                         try:
                             output_reward_func = reward_func(
                                 prompts=filtered_reward_prompts, # Q+A1+I context
                                 completions=filtered_reward_completions, # A2
                                 first_completions=filtered_completions_text_a1, # Pass A1 text
                                 **reward_kwargs
                             )
                             temp_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                         except Exception as e:
                             print(f"Error during reward function {reward_func_name}: {e}")
                             # temp_rewards_per_func[:, i] remains zero

             # Scatter results back to the original size tensor
             rewards_per_func = torch.zeros(num_local_prompts, len(self.reward_funcs), device=device)
             rewards_per_func[torch.tensor(valid_reward_indices, device=device)] = temp_rewards_per_func


        # --- Calculate Advantages (Requires Gathering Rewards) ---
        print(f"here13: Calculating Advantages. Process {self.accelerator.process_index}")
        gathered_rewards_per_func = gather(rewards_per_func) # (B_total*G, num_rewards)

        # Compute combined reward score (on gathered tensor)
        gathered_rewards = (gathered_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1) # (B_total*G,)

        # Compute grouped-wise rewards
        num_prompts_total = num_total_prompts // self.num_generations
        # Handle case where num_total_prompts might not be perfectly divisible if errors occurred
        if num_total_prompts % self.num_generations != 0:
             print(f"Warning: Total prompts ({num_total_prompts}) not divisible by num_generations ({self.num_generations}). Advantage calculation might be skewed.")
             # Adjust num_prompts_total or pad gathered_rewards if necessary
             expected_size = num_prompts_total * self.num_generations
             gathered_rewards = gathered_rewards[:expected_size] # Truncate for now

        # Reshape requires contiguous tensor
        gathered_rewards = gathered_rewards.contiguous()
        mean_grouped_rewards = gathered_rewards.view(num_prompts_total, self.num_generations).mean(dim=1)
        std_grouped_rewards = gathered_rewards.view(num_prompts_total, self.num_generations).std(dim=1)

        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B_total*G,)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B_total*G,)
        # Ensure advantages match shape of (potentially truncated) gathered_rewards
        gathered_advantages = gathered_rewards - mean_grouped_rewards[:gathered_rewards.size(0)]
        if self.args.scale_rewards:
            gathered_advantages = gathered_advantages / (std_grouped_rewards[:gathered_rewards.size(0)] + 1e-4)

        # Get the advantages corresponding to the local process slice
        # Use original slice definition based on num_local_prompts
        local_slice_indices = torch.arange(process_slice.start, process_slice.stop)
        # Select advantages corresponding to local indices from the potentially truncated gathered_advantages
        advantages = gathered_advantages[local_slice_indices] # (local_B*G,)


        # --- Logging ---
        print(f"here14: Logging results. Process {self.accelerator.process_index}")
        mode = "eval" if self.control.should_evaluate else "train"
        # Gather lengths for accurate mean calculation (use combined mask)
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length) # Logs total length A1+A2

        # Log mean reward per function across all processes
        reward_per_func_mean = gathered_rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module): reward_func_name = f"reward_{i}"
            else: reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        # Log overall mean reward and mean std dev across groups
        self._metrics[mode]["reward"].append(gathered_rewards.mean().item())
        # Calculate std dev mean based on valid groups
        mean_reward_std = gathered_rewards.view(num_prompts_total, self.num_generations).std(dim=1).mean().item()
        self._metrics[mode]["reward_std"].append(mean_reward_std)

        # Log completions (Gather all necessary text)
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
             # Gather original Q prompts, A1 text, A2 text, and A2 prompts for context
             all_prompts_q_text = gather_object(prompts_text)
             all_completions_text_a1 = gather_object(local_completions_text_a1_list)
             all_prompts_text_turn2 = gather_object(local_prompts_text_turn2) # Gather A2 prompts
             all_completions_text_a2 = gather_object(local_completions_text_a2_list)
             # Use gathered rewards
             all_rewards_to_log = gathered_rewards.tolist()

             if self.accelerator.is_main_process:
                #  print(f"\n--- Step {self.state.global_step} Two-Turn Completions (1:n:n) ---")
                #  num_to_log = min(5 * self.num_generations, len(all_prompts_q_text)) # Log more to see variations
                #  for i in range(num_to_log):
                #       print(f"\n[Example {i+1} (Q prompt {i//self.num_generations + 1}, Gen {(i % self.num_generations) + 1})]")
                #       print(f"  Prompt (Q): {all_prompts_q_text[i]}")
                #       print(f"  Turn 1 (A1): {all_completions_text_a1[i]}")
                #       # print(f"  Prompt (A2): {all_prompts_text_turn2[i]}") # Optional: Log A2 prompt
                #       print(f"  Turn 2 (A2): {all_completions_text_a2[i]}")
                #       # Ensure reward index is valid
                #       reward_val = all_rewards_to_log[i] if i < len(all_rewards_to_log) else float('nan')
                #       print(f"  Reward: {reward_val:.4f}")
                #  print("------------------------------------\n")

                 if self.args.report_to and "wandb" in self.args.report_to and wandb and wandb.run:
                     if pd:
                         table_data = {
                             "step": [str(self.state.global_step)] * len(all_rewards_to_log),
                             "prompt_q": all_prompts_q_text[:len(all_rewards_to_log)], # Slice to match rewards
                             "completion_a1": all_completions_text_a1[:len(all_rewards_to_log)],
                             "prompt_a2": all_prompts_text_turn2[:len(all_rewards_to_log)],
                             "completion_a2": all_completions_text_a2[:len(all_rewards_to_log)],
                             "reward": all_rewards_to_log,
                         }
                         # Ensure all lists have the same length before creating DataFrame
                         min_len = min(len(v) for v in table_data.values())
                         table_data_truncated = {k: v[:min_len] for k, v in table_data.items()}
                         try:
                             df = pd.DataFrame(table_data_truncated)
                             wandb.log({"two_turn_completions_1nn": wandb.Table(dataframe=df)})
                         except Exception as e:
                             print(f"Error creating wandb table: {e}")
                     else:
                         print("Pandas not available for wandb table logging.")


        # --- Return results for loss computation ---
        print(f"here15: Returning results for loss. Process {self.accelerator.process_index}")
        # Return dictionary structured for compute_loss expecting combined A1+A2
        return {
            "prompt_ids": prompt_ids,              # Original Q prompt IDs (local_B*G, P)
            "prompt_mask": prompt_mask,             # Original Q prompt mask (local_B*G, P)
            "completion_ids": completion_ids,       # Combined A1+A2 IDs (local_B*G, C1+C2)
            "completion_mask": completion_mask,     # Combined A1+A2 mask (local_B*G, C1+C2)
            "old_per_token_logps": old_per_token_logps, # Combined A1+A2 logps (local_B*G, C1+C2)
            "ref_per_token_logps": ref_per_token_logps, # Combined A1+A2 ref logps (local_B*G, C1+C2)
            "advantages": advantages,               # Local advantages slice (local_B*G,)
        }