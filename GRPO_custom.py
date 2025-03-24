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



class VerificationGRPOTrainer(GRPOTrainer):
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
                final_second_turn_prompts = []
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



class SwitchingGRPOTrainer(GRPOTrainer):

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
                final_second_turn_prompts = []
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

            
            else:
                # Non-main processes get placeholders.
                final_second_turn_prompts = [None] * len(all_prompts_text)
                completion_ids_list = [None] * len(all_prompts_text)
            
            # 2.6) Broadcast the final completion_ids to every process. Then slice out only the portion that belongs to this process.
            final_second_turn_prompts = broadcast_object_list(final_second_turn_prompts, from_process=0)
            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)

            # Now we take the local slice to keep shape consistent with the local batch.
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids_list = completion_ids_list[process_slice]
            # 2.7) Convert to a padded tensor for logprob calculations:
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

            # The "prompt" we need to consider for scoring is now the second-turn prompt, 
            # so we tokenize that. We already have it in final_second_turn_prompts => we slice the relevant piece for this process.
            local_second_turn_prompts = final_second_turn_prompts[process_slice]
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
            
            # Potentially truncate
            if self.max_prompt_length is not None:
                second_prompt_ids = second_prompt_ids[:, -self.max_prompt_length:]
                second_prompt_mask = second_prompt_mask[:, -self.max_prompt_length:]

            # The final “prompt + completion” is this second-turn prompt + A2.
            prompt_completion_ids = torch.cat([second_prompt_ids, completion_ids], dim=1)
        
        else:
            print("USE VLLM!")


        # Mask everything after the first EOS token
        is_eos = completion_ids.eq(self.processing_class.eos_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]

        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask to get the attention_mask for the entire sequence.
        attention_mask = torch.cat([second_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)


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

        # 5) Decode the final completions for reward function usage
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    
        # If your data is conversational, adapt accordingly. We keep the “standard text” scenario:
        completions = completions_text

        # ================================
        # 6) Compute rewards from your configured reward functions (unchanged):
        #    We accumulate rewards_per_func, sum them up with self.reward_weights, gather, etc.
        #    The code below is basically the same as the original single-turn approach.
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
        # Collect global rewards
        rewards_per_func = gather(rewards_per_func)
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


    # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
    # second row shows the second sampled batch, and so on.
    #
    #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
    #
    #               global_step   step     <───────>  num_generations=3
    #                                      <───────────> per_device_train_batch_size=4
    #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
    #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
    #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
    #
    #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
    #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
    #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
    #
    #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
    #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
    #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
    #       

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                if self.state.global_step % 2 == 1:
                    print("2 TURN TRAINING")
                    inputs = self._generate_and_score_completions(inputs)
                    self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
                else:
                    print("1 TURN TRAINING")
                    inputs = super()._generate_and_score_completions(inputs)
                    self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = super()._generate_and_score_completions(inputs)
        return inputs
    
