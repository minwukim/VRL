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


def profiling_decorator(func: callable) -> callable:
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    Args:
        func (`callable`):
            Function to be profiled.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_decorator

    class MyTrainer(Trainer):
        @profiling_decorator
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            # Code to profile: simulate a computationally expensive operation
            result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


def extract_math_prompt(conversation_str):
    """
    Extracts the math prompt from a conversation string.
    
    The expected format is:
      "…tag.
      <math prompt (possibly spanning multiple lines)>
      
      Assistant: <think>"
      
    This function returns the math prompt text between the final newline after the instructions and before the
    "Assistant: <think>" marker.
    """
    # The regex looks for a newline, then captures any text (including newlines) lazily up to a newline
    # that precedes "Assistant:" and "<think>".
    pattern = r'\n(.*?)\n\s*Assistant:\s*<think>'
    match = re.search(pattern, conversation_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_user_text(text):
    """
    Extracts text from the user block, ignoring any preceding system instructions.
    It looks for text between '<|im_start|>user' and '<|im_end|>' markers.
    """
    match = re.search(r"<\|im_start\|>user(.*?)<\|im_end\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def remove_tokens(text):
    """
    Removes unnecessary tokens such as '<|im_start|>assistant', '<|im_start|>system', and '<|im_end|>'.
    """
    tokens = ["<|im_start|>assistant", "<|im_start|>system", "<|im_end|>"]
    for token in tokens:
        text = text.replace(token, "")
    return text.strip()

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
    # if not is_rich_available():
    #     raise ImportError("This feature requires `rich` to be installed. Please install it first: `pip install rich`")

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):
        table.add_row(Text(prompt), Text(completion), f"{reward:.2f}")  # Formatting reward to 2 decimal places
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)

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

def extract_a1_text(prompt_string):
    """
    Extracts a1_text from a formatted prompt string.

    Args:
        prompt_string (str): The full prompt string.

    Returns:
        str or None: The extracted a1_text, or None if not found.
    """
    pattern = r"Response:\s*<think>\s*(.*?)\s*Assistant:\s*<think>"
    match = re.search(pattern, prompt_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None




class TwoTurnGRPOTrainer2(GRPOTrainer):

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
    
        # 1. Preprocess the original prompt (Q)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self,prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            
            # Gather prompts from all processes -> one big list on rank 0
            all_prompts_text = gather_object(prompts_text)

            if self.accelerator.is_main_process:            
                # 2.1) De-duplicate the original prompts: 
                # for a batch of size B = (#distinct Q * self.num_generations), we want the distinct Q.
                ordered_unique_prompts = all_prompts_text[:: self.num_generations]

                sampling_params_1n = deepcopy(self.sampling_params)
                sampling_params_1n.n = 1

                with profiling_context(self, "vLLM.generate (first turn)"):
                    first_turn_outputs = self.llm.generate(
                        ordered_unique_prompts, 
                        sampling_params=sampling_params_1n,
                        use_tqdm=False,
                    )

                # Each item in first_turn_outputs has exactly 1 completion
                first_turn_completions_ids = [o.outputs[0].token_ids for o in first_turn_outputs]
                # Decode the first_turn outputs to string.
                first_turn_completions_text = [self.processing_class.decode(ids, skip_special_tokens=True) for ids in first_turn_completions_ids]

                # 2.3) Build the new prompt
                added_instruction = (
                    "A conversation between User and Assistant. Given a question and a corresponding response provided below, the Assistant systematically reviews and explains each step of the reasoning process to verify the correctness of the response."
                    "If errors are found, the Assistant identifies and corrects them, then re-solves the problem. If the response is correct, the Assistant confirms it and returns the same final answer."
                    "The assistant first thinks about the reasoning process in mind, including verification, correction, and resolving the problem if necessary. Then provides the user with the answer."
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>." 
                    "The reasoning process, including verification and correction, is enclosed within <think> </think> tags, while the final solution is enclosed within <answer> </answer> tags. The final answer is formatted within \\boxed{{}} to enable direct extraction for grading."
                    "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag."
                )

                second_turn_prompts = []
                for q_text, a1_text in zip(ordered_unique_prompts, first_turn_completions_text):

                    # Build the second-turn prompt
                    example = {
                        "prompt": (
                            added_instruction
                            + "\n\nQuestion:\n" + extract_math_prompt(q_text)
                            + "\n\nResponse:\n<think> " + a1_text
                            + "\nAssistant: <think>"  # continuing the chain-of-thought
                        ),
                    }
                    second_turn_prompts.append(maybe_apply_chat_template(example, self.processing_class)["prompt"])
                
                # 2.4) Re-duplicate the second-turn prompts so each distinct prompt is repeated G times,
                #      exactly matching the shape of the original repeated batch. 
                #      If we have N distinct Q, each will produce N distinct second-turn prompts, each repeated G times => total = N*G
                for new_p in second_turn_prompts:
                    final_second_turn_prompts.extend([new_p] * self.num_generations)

                # 2.5) Now generate the second turn (A2). 
                #      We have len(ordered_unique_prompts)*self.num_generations prompts in final_second_turn_prompts.
                #      Each second-turn prompt will produce exactly 1 completion because we want G = self.num_generations distinct completions overall.

                with profiling_context(self, "vLLM.generate (second turn)"):
                    second_turn_outputs = self.llm.generate(
                        final_second_turn_prompts,
                        sampling_params=sampling_params_1n,
                        use_tqdm=False,
                    )
                # Extract final completions (A2).
                completion_ids_list = [out.outputs[0].token_ids for out in second_turn_outputs]
                print("MAIN PROCESS COMPLETION IDS:", len(completion_ids_list))

            
            else:
                # Non-main processes get placeholders.
                final_second_turn_prompts = [None] * len(all_prompts_text)
                completion_ids_list = [None] * len(all_prompts_text)
            
            # 2.6) Broadcast the final completion_ids to every process. Then slice out only the portion that belongs to this process.
            print("-------------------------------------------------")
            print(self.accelerator.is_main_process, "broadcast second turn prompts")
            final_second_turn_prompts = broadcast_object_list(final_second_turn_prompts, from_process=0)
            print(self.accelerator.is_main_process, "broadcast second turn prompts DONE", print(final_second_turn_prompts[0]))
            print("-------------------------------------------------")
            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)

            # Now we take the local slice to keep shape consistent with the local batch.
            print(self.accelerator.is_main_process,"HERE1")
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids_list = completion_ids_list[process_slice]
            print(self.accelerator.is_main_process,"HERE2")
            # 2.7) Convert to a padded tensor for logprob calculations:
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

            # The "prompt" we need to consider for scoring is now the second-turn prompt, 
            # so we tokenize that. We already have it in final_second_turn_prompts => we slice the relevant piece for this process.
            print(self.accelerator.is_main_process,"HERE3")
            local_second_turn_prompts = final_second_turn_prompts[process_slice]
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            print(self.accelerator.is_main_process, "local_second_turn_prompts",local_second_turn_prompts)
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            second_prompt_inputs = self.processing_class(
                text=local_second_turn_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            second_prompt_inputs = Trainer._prepare_inputs(self,second_prompt_inputs)
            second_prompt_ids = second_prompt_inputs["input_ids"].to(device)
            second_prompt_mask = second_prompt_inputs["attention_mask"].to(device)
            
            print(self.accelerator.is_main_process,"HERE4")
            # Potentially truncate
            if self.max_prompt_length is not None:
                second_prompt_ids = second_prompt_ids[:, -self.max_prompt_length:]
                second_prompt_mask = second_prompt_mask[:, -self.max_prompt_length:]

            # The final “prompt + completion” is this second-turn prompt + A2.
            prompt_completion_ids = torch.cat([second_prompt_ids, completion_ids], dim=1)

            print(self.accelerator.is_main_process,"HERE5")
        
        else:
            print("USE VLLM!")


        # Mask everything after the first EOS token
        print("HERE6")
        is_eos = completion_ids.eq(self.processing_class.eos_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        print(self.accelerator.is_main_process,"HERE7")

        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        print(self.accelerator.is_main_process,"HERE8")
        # Concatenate prompt_mask with completion_mask to get the attention_mask for the entire sequence.
        attention_mask = torch.cat([second_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        print(self.accelerator.is_main_process,"HERE9")

        # 3) If we do multiple updates (num_iterations>1), we also need old_per_token_logps. 
        #    Otherwise, we can skip that (same as your original code).
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            # 4) Reference model logprobs
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

        print(self.accelerator.is_main_process,"HERE10")
        # 5) Decode the final completions for reward function usage
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    
        # If your data is conversational, adapt accordingly. We keep the “standard text” scenario:
        completions = completions_text

        # ================================
        print(self.accelerator.is_main_process,"HERE11")
        # 6) Compute rewards from your configured reward functions (unchanged):
        #    We accumulate rewards_per_func, sum them up with self.reward_weights, gather, etc.
        #    The code below is basically the same as the original single-turn approach.
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        print(self.accelerator.is_main_process,"HERE12")
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__

            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(final_second_turn_prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(final_second_turn_prompts, completions)]

                    reward_inputs = reward_processing_class(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(
                        prompts=final_second_turn_prompts,
                        completions=completions,
                        **reward_kwargs
                    )
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func,
                        dtype=torch.float32,
                        device=device
                    )
        print(self.accelerator.is_main_process,"here21")
        # Collect global rewards
        rewards_per_func = gather(rewards_per_func)
        print(self.accelerator.is_main_process,"here22")
        # Weighted sum over reward functions
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only local portion
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]


        # ------------------
        # 9. Logging
        # ------------------
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(local_second_turn_prompts)
            print("prompts_to_log", len(prompts_to_log))
            completions_to_log = gather_object(completions_text)
            print("completions_to_log", len(prompts_to_log))
            rewards_to_log = rewards.tolist()
            print("rewards_to_log", len(rewards_to_log))

            if self.accelerator.is_main_process:
                # You could optionally do a pretty print here
                # if is_rich_available():
                print_prompt_completions_sample(
                    prompts_to_log,
                    completions_to_log,
                    rewards_to_log,
                    self.state.global_step,
                )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards_to_log,
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # ------------------
        # 10. Return final dictionary
        # ------------------
        return {
            "prompt_ids": second_prompt_ids,
            "prompt_mask": second_prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

import torch
import torch.nn as nn
from typing import Union, Any, Dict, List, Optional
from trl import GRPOTrainer
from trl.trainer.utils import (
    pad,
    gather_object,
    broadcast_object_list,
    unwrap_model_for_generation,
    is_conversational,
    maybe_apply_chat_template,
    apply_chat_template, # Added import
    profiling_context, # Added import
)
from trl.trainer.log import is_rich_available # Added import
if is_rich_available():
    from trl.trainer.log import print_prompt_completions_sample # Added import
try:
    import wandb # Added import
except ImportError:
    wandb = None
from copy import deepcopy # Added import
from accelerate.utils import gather # Added import

# Assume necessary imports from transformers are available
# from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig, StoppingCriteriaList

class TwoTurnGRPOTrainer(GRPOTrainer):
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
        # Tokenize the instruction once using the main tokenizer for efficiency
        # We assume instruction doesn't need special tokens prepended usually,
        # but adjust if needed based on your model/tokenizer.
        instruction_tokens = self.processing_class(
            self.correction_instruction,
            return_tensors="pt",
            add_special_tokens=False, # Usually False for mid-sequence insertion
        )
        self.instruction_ids = instruction_tokens["input_ids"].squeeze(0) # Remove batch dim
        self.instruction_mask = instruction_tokens["attention_mask"].squeeze(0)
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
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # Original prompts (Q)
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # --- Prepare Initial Prompt Inputs (Q) ---
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"] # (B*G, P)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # --- Turn 1 Generation (Q -> A1) ---
        # Uses self.sampling_params with n=num_generations
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                with profiling_context(self, "vLLM.generate_A1"):
                    all_outputs_a1 = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                completion_ids_a1_list = []
                for outputs in all_outputs_a1:
                    for output in outputs.outputs:
                        completion_ids_a1_list.append(output.token_ids)
            else:
                completion_ids_a1_list = [None] * len(all_prompts_text)

            completion_ids_a1_list = broadcast_object_list(completion_ids_a1_list, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids_a1_list = completion_ids_a1_list[process_slice]

            completion_ids_a1 = [torch.tensor(ids, device=device) for ids in completion_ids_a1_list]
            completion_ids_a1 = pad(completion_ids_a1, padding_value=self.processing_class.pad_token_id, padding_side="right") # (B*G, C1)
            # Mask A1 after EOS
            is_eos_a1 = completion_ids_a1 == self.processing_class.eos_token_id
            eos_idx_a1 = torch.full((is_eos_a1.size(0),), is_eos_a1.size(1), dtype=torch.long, device=device)
            eos_idx_a1[is_eos_a1.any(dim=1)] = is_eos_a1.int().argmax(dim=1)[is_eos_a1.any(dim=1)]
            sequence_indices_a1 = torch.arange(is_eos_a1.size(1), device=device).expand(is_eos_a1.size(0), -1)
            completion_mask_a1 = (sequence_indices_a1 <= eos_idx_a1.unsqueeze(1)).int() * (completion_ids_a1 != self.processing_class.pad_token_id).int() # (B*G, C1)

        else: # Regular HF Generation for A1
             with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                # Note: HF generate might not directly support n>1 in the same way vLLM does per prompt easily.
                # This path might need adjustment if num_generations > 1 is critical.
                # Assuming num_generations=1 or handled by input duplication for this branch.
                if self.num_generations > 1:
                     print("Warning: Standard HF generation path in TwoTurnGRPO might not correctly handle num_generations > 1 like vLLM. Consider using vLLM.")

                prompt_completion_ids_a1 = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )
                prompt_length = prompt_ids.size(1)
                # prompt_ids = prompt_completion_ids_a1[:, :prompt_length] # Already have original prompt_ids
                completion_ids_a1 = prompt_completion_ids_a1[:, prompt_length:] # (B*G, C1)
                # Mask A1 after EOS
                is_eos_a1 = completion_ids_a1 == self.processing_class.eos_token_id
                eos_idx_a1 = torch.full((is_eos_a1.size(0),), is_eos_a1.size(1), dtype=torch.long, device=device)
                eos_idx_a1[is_eos_a1.any(dim=1)] = is_eos_a1.int().argmax(dim=1)[is_eos_a1.any(dim=1)]
                sequence_indices_a1 = torch.arange(is_eos_a1.size(1), device=device).expand(is_eos_a1.size(0), -1)
                completion_mask_a1 = (sequence_indices_a1 <= eos_idx_a1.unsqueeze(1)).int() * (completion_ids_a1 != self.processing_class.pad_token_id).int() # (B*G, C1)


        # Decode A1 for constructing Turn 2 prompts
        # Apply mask before decoding to avoid including EOS and padding
        completions_text_a1_list = []
        for ids, mask in zip(completion_ids_a1, completion_mask_a1):
             masked_ids = ids[mask.bool()]
             completions_text_a1_list.append(self.processing_class.decode(masked_ids, skip_special_tokens=True))


        # --- Prepare Turn 2 Inputs (Q + A1 + I) ---
        # Combine original prompt text, A1 text, and the instruction text
        # Handling chat templates here could be complex, requires careful formatting.
        # Sticking to simpler text concatenation for now.
        prompts_text_turn2 = [
            p_text + a1_text + self.correction_instruction
            for p_text, a1_text in zip(prompts_text, completions_text_a1_list)
        ]

        # Tokenize the combined prompts Q+A1+I for Turn 2 generation
        prompt_inputs_turn2 = self.processing_class(
            prompts_text_turn2, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs_turn2 = super()._prepare_inputs(prompt_inputs_turn2)
        prompt_ids_turn2, prompt_mask_turn2 = prompt_inputs_turn2["input_ids"], prompt_inputs_turn2["attention_mask"] # (B*G, P+C1+I_len)

        # --- Turn 2 Generation ((Q + A1 + I) -> A2) ---
        # **Crucially, generate only ONE completion (n=1) for each Q+A1+I input.**
        if self.args.use_vllm:
            # Create sampling params for n=1 generation
            sampling_params_turn2 = deepcopy(self.sampling_params)
            sampling_params_turn2.n = 1

            all_prompts_text_turn2 = gather_object(prompts_text_turn2)
            if self.accelerator.is_main_process:
                 # No need to dedupe here, each Q+A1+I should be unique enough
                 with profiling_context(self, "vLLM.generate_A2"):
                      all_outputs_a2 = self.llm.generate(
                           all_prompts_text_turn2, sampling_params=sampling_params_turn2, use_tqdm=False
                      )
                 # Since n=1, each prompt yields one output sequence
                 completion_ids_a2_list = [output.outputs[0].token_ids for output in all_outputs_a2]
            else:
                completion_ids_a2_list = [None] * len(all_prompts_text_turn2)

            completion_ids_a2_list = broadcast_object_list(completion_ids_a2_list, from_process=0)
            completion_ids_a2_list = completion_ids_a2_list[process_slice] # Get local slice

            completion_ids_a2 = [torch.tensor(ids, device=device) for ids in completion_ids_a2_list]
            completion_ids_a2 = pad(completion_ids_a2, padding_value=self.processing_class.pad_token_id, padding_side="right") # (B*G, C2)

            # Mask A2 after EOS
            is_eos_a2 = completion_ids_a2 == self.processing_class.eos_token_id
            eos_idx_a2 = torch.full((is_eos_a2.size(0),), is_eos_a2.size(1), dtype=torch.long, device=device)
            eos_idx_a2[is_eos_a2.any(dim=1)] = is_eos_a2.int().argmax(dim=1)[is_eos_a2.any(dim=1)]
            sequence_indices_a2 = torch.arange(is_eos_a2.size(1), device=device).expand(is_eos_a2.size(0), -1)
            completion_mask_a2 = (sequence_indices_a2 <= eos_idx_a2.unsqueeze(1)).int() * (completion_ids_a2 != self.processing_class.pad_token_id).int() # (B*G, C2)


        else: # Regular HF Generation for A2
            # Ensure generation config produces only 1 sequence if not already set
            generation_config_turn2 = deepcopy(self.generation_config)
            generation_config_turn2.num_return_sequences = 1
            generation_config_turn2.num_beams = 1 # Ensure greedy or sampling, not beam search returning multiple

            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                prompt_completion_ids_a2 = unwrapped_model.generate(
                    prompt_ids_turn2, attention_mask=prompt_mask_turn2, generation_config=generation_config_turn2
                )
                prompt_length_turn2 = prompt_ids_turn2.size(1)
                completion_ids_a2 = prompt_completion_ids_a2[:, prompt_length_turn2:] # (B*G, C2)

                # Mask A2 after EOS
                is_eos_a2 = completion_ids_a2 == self.processing_class.eos_token_id
                eos_idx_a2 = torch.full((is_eos_a2.size(0),), is_eos_a2.size(1), dtype=torch.long, device=device)
                eos_idx_a2[is_eos_a2.any(dim=1)] = is_eos_a2.int().argmax(dim=1)[is_eos_a2.any(dim=1)]
                sequence_indices_a2 = torch.arange(is_eos_a2.size(1), device=device).expand(is_eos_a2.size(0), -1)
                completion_mask_a2 = (sequence_indices_a2 <= eos_idx_a2.unsqueeze(1)).int() * (completion_ids_a2 != self.processing_class.pad_token_id).int() # (B*G, C2)


        # --- Combine Sequences and Calculate Log Probabilities ---
        # We need logps for A1 (given Q) and A2 (given Q+A1+I)

        # Sequence for A1 logps: Q + A1
        input_ids_a1 = torch.cat([prompt_ids, completion_ids_a1], dim=1) # (B*G, P + C1)
        attention_mask_a1 = torch.cat([prompt_mask, completion_mask_a1], dim=1) # (B*G, P + C1)
        logits_to_keep_a1 = completion_ids_a1.size(1)

        # Sequence for A2 logps: Q + A1 + I + A2 (using turn2 prompt which is Q+A1+I)
        input_ids_a2 = torch.cat([prompt_ids_turn2, completion_ids_a2], dim=1) # (B*G, P+C1+I + C2)
        attention_mask_a2 = torch.cat([prompt_mask_turn2, completion_mask_a2], dim=1) # (B*G, P+C1+I + C2)
        logits_to_keep_a2 = completion_ids_a2.size(1)

        with torch.no_grad():
            # Calculate necessary logps for A1
            if self.num_iterations > 1:
                old_per_token_logps_a1 = self._get_per_token_logps(
                    self.model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                ) # (B*G, C1)
            else:
                old_per_token_logps_a1 = None # Will use detached current logps later

            if self.beta == 0.0:
                ref_per_token_logps_a1 = None
            elif self.ref_model is not None:
                ref_per_token_logps_a1 = self._get_per_token_logps(
                    self.ref_model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                ) # (B*G, C1)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps_a1 = self._get_per_token_logps(
                        self.model, input_ids_a1, attention_mask_a1, logits_to_keep_a1
                    ) # (B*G, C1)

            # Calculate necessary logps for A2
            if self.num_iterations > 1:
                 # Need the same model state as A1 generation for consistency
                 # This assumes model state *didn't* change between A1/A2 logp calcs
                 # If using iteration > 1, this might need careful state management
                 # or recomputation if model could change. Sticking to original logic:
                old_per_token_logps_a2 = self._get_per_token_logps(
                    self.model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                ) # (B*G, C2)
            else:
                 old_per_token_logps_a2 = None # Will use detached current logps later

            if self.beta == 0.0:
                ref_per_token_logps_a2 = None
            elif self.ref_model is not None:
                ref_per_token_logps_a2 = self._get_per_token_logps(
                    self.ref_model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                ) # (B*G, C2)
            else:
                 with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps_a2 = self._get_per_token_logps(
                        self.model, input_ids_a2, attention_mask_a2, logits_to_keep_a2
                    ) # (B*G, C2)

        # Combine logps (pad and concatenate) - needed later for loss calculation input
        # We need to ensure they have the same sequence length before concatenating logps.
        # The loss function operates on a combined sequence, so we create combined tensors.
        # This assumes the loss function (`compute_loss`) will correctly use the combined mask.

        # Create combined completion sequence and mask (A1 + A2) for loss input structure
        # NOTE: This combination is *conceptual* for feeding the loss function.
        # The logps were calculated separately based on their correct contexts.
        # We need padding between A1 and A2 if their lengths differ. Let's pad A1 to max_a1_len and A2 to max_a2_len
        # Then concatenate. The mask handles the effective parts.

        max_len_c1 = completion_ids_a1.size(1)
        max_len_c2 = completion_ids_a2.size(1)

        # Combine IDs: Pad A1 and A2 to their max lengths, then concat
        completion_ids = torch.cat(
             (completion_ids_a1, completion_ids_a2), dim=1
        ) # (B*G, C1+C2)

        # Combine Masks: Concat masks
        completion_mask = torch.cat(
            (completion_mask_a1, completion_mask_a2), dim=1
        ) # (B*G, C1+C2)

        # Combine Logps (handling None cases)
        def _combine_logps(logps1, logps2, mask1, mask2):
            if logps1 is None and logps2 is None:
                return None
            # Pad logps to match max lengths before concat (use 0 where mask is 0)
            padded_logps1 = torch.zeros_like(completion_ids_a1, dtype=torch.float)
            if logps1 is not None:
                padded_logps1[:, :logps1.size(1)] = logps1 * mask1 # Mask out padding logps
            padded_logps2 = torch.zeros_like(completion_ids_a2, dtype=torch.float)
            if logps2 is not None:
                padded_logps2[:, :logps2.size(1)] = logps2 * mask2 # Mask out padding logps
            return torch.cat((padded_logps1, padded_logps2), dim=1)


        old_per_token_logps = _combine_logps(old_per_token_logps_a1, old_per_token_logps_a2, completion_mask_a1, completion_mask_a2)
        ref_per_token_logps = _combine_logps(ref_per_token_logps_a1, ref_per_token_logps_a2, completion_mask_a1, completion_mask_a2)


        # --- Decode Final Completions (A2) and Calculate Rewards ---
        # Reward is based on the final answer A2, potentially considering Q and A1+I as context.
        # Decode A2 for reward calculation and logging
        completions_text_a2_list = []
        for ids, mask in zip(completion_ids_a2, completion_mask_a2):
             masked_ids = ids[mask.bool()]
             completions_text_a2_list.append(self.processing_class.decode(masked_ids, skip_special_tokens=True))

        # Prepare inputs for reward functions based on A2
        # The 'prompts' argument to reward_func should arguably be the context for A2 (Q+A1+I)
        # And 'completions' should be A2.
        # Alternatively, if reward func expects original prompt Q and *full* completion A1+I+A2, adjust accordingly.
        # Assuming reward function needs context (Q+A1+I) and final completion (A2):
        reward_prompts = prompts_text_turn2 # Q+A1+I
        reward_completions = completions_text_a2_list # A2

        # If conversational, format appropriately. This part might need careful adjustment
        # based on how your reward model expects multi-turn conversations.
        # Here's a guess assuming the reward model wants the full history:
        if is_conversational(inputs[0]):
            # Construct full conversation history Q -> A1 -> I -> A2 for reward model
            full_conversations = []
            for q_list, a1_text, a2_text in zip(prompts, completions_text_a1_list, completions_text_a2_list):
                # Start with original prompt messages
                conv = deepcopy(q_list)
                # Append A1
                conv.append({"role": "assistant", "content": a1_text})
                # Append I (as user instruction?) - This is ambiguous, maybe skip adding I explicitly
                # Or treat I implicitly as the prompt leading to A2
                # Append A2
                conv.append({"role": "assistant", "content": a2_text}) # A2 generated from context Q+A1+I
                full_conversations.append(conv)

            # How reward models process this depends heavily on their training data format
            # Option 1: Give full history to reward model's chat template
            reward_inputs_texts = []
            for conv in full_conversations:
                 # Assume reward processing class has a chat template
                 # This assumes the reward tokenizer/template handles the full conversation
                 processed = apply_chat_template({"messages": conv}, self.reward_processing_classes[0]) # Use first reward tokenizer as example
                 reward_inputs_texts.append(processed["text"])

            # Option 2: Give context (Q+A1+I) and completion (A2) separately
            # reward_prompts = prompts_text_turn2
            # reward_completions = completions_text_a2_list
            # texts = [p + c for p, c in zip(reward_prompts, reward_completions)] # Requires reward model trained on this format

        else: # Non-conversational
            # Combine context (Q+A1+I) and completion (A2) for reward model
            reward_inputs_texts = [p + c for p, c in zip(reward_prompts, reward_completions)]


        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__

            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                     # Use the prepared texts (either from chat template or concatenation)
                     reward_inputs = reward_processing_class(
                         reward_inputs_texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False # Check add_special_tokens
                     )
                     reward_inputs = super()._prepare_inputs(reward_inputs)
                     with torch.inference_mode():
                         # Assuming reward model outputs score in logits[:, 0]
                         rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else: # Function-based reward
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    # Pass relevant info. The function needs to know how to handle the turns.
                    # Option: Pass Q, A1, I, A2 explicitly if needed by the function.
                    # Here, passing the final context and completion:
                    output_reward_func = reward_func(prompts=reward_prompts, completions=reward_completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # --- Calculate Advantages and Log Metrics ---
        rewards_per_func = gather(rewards_per_func) # Gather rewards (B*G, num_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1) # (B*G,)

        # Compute grouped-wise rewards (original GRPO logic)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4) # (B*G,)

        # Slice advantages to keep only local process data
        advantages = advantages[process_slice] # (local_B*G,)

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        # Use the combined completion mask length (A1 + A2) for logging
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length) # Logs total length A1+A2

        reward_per_func_mean = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            # Name extraction logic from original code
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # Log mean std dev across groups

        # Log completions (Log Q, A1, and A2 for clarity)
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
             prompts_to_log = gather_object(prompts_text) # Original Q
             completions_a1_to_log = gather_object(completions_text_a1_list) # A1
             completions_a2_to_log = gather_object(completions_text_a2_list) # A2
             rewards_to_log = rewards.tolist() # Final rewards

             if self.accelerator.is_main_process:
                 # Modified logging to show Q, A1, A2
                 print(f"\n--- Step {self.state.global_step} Two-Turn Completions ---")
                 for i in range(min(5, len(prompts_to_log))): # Log first few examples
                      print(f"\n[Example {i+1}]")
                      print(f"  Prompt (Q): {prompts_to_log[i]}")
                      print(f"  Turn 1 (A1): {completions_a1_to_log[i]}")
                      # print(f"  Instruction (I): {self.correction_instruction}") # Optional: print instruction
                      print(f"  Turn 2 (A2): {completions_a2_to_log[i]}")
                      print(f"  Reward: {rewards_to_log[i]:.4f}")
                 print("------------------------------------\n")

                 if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                     import pandas as pd
                     table = {
                         "step": [str(self.state.global_step)] * len(rewards_to_log),
                         "prompt": prompts_to_log,
                         "completion_A1": completions_a1_to_log,
                         "completion_A2": completions_a2_to_log,
                         "reward": rewards_to_log,
                     }
                     df = pd.DataFrame(table)
                     wandb.log({"two_turn_completions": wandb.Table(dataframe=df)})


        # --- Return results for loss computation ---
        # Return combined sequences and masks, original prompt info, logps, and advantages.
        return {
            "prompt_ids": prompt_ids, # (B*G, P)
            "prompt_mask": prompt_mask, # (B*G, P)
            "completion_ids": completion_ids, # (B*G, C1+C2) - Combined A1+A2 IDs
            "completion_mask": completion_mask, # (B*G, C1+C2) - Combined A1+A2 mask (ignores instruction I implicitly)
            "old_per_token_logps": old_per_token_logps, # (B*G, C1+C2) - Combined A1+A2 logps
            "ref_per_token_logps": ref_per_token_logps, # (B*G, C1+C2) - Combined A1+A2 ref logps
            "advantages": advantages, # (local_B*G,) - Single advantage value per trajectory
        }