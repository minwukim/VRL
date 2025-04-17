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

from math_verify import verify, parse

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


def check_correctness(completion: str, ground_truth: str) -> float:
    """
    Directly computes the correctness regardless of formatting.
    Returns 1.0 if the parsed completion matches the ground truth; otherwise 0.0.
    """
    return 1.0 if verify(parse(completion), parse(ground_truth)) else 0.0



class OON_Oracle_GRPOTrainer(GRPOTrainer):

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
            all_ground_truths = gather_object([x["answer"] for x in inputs])

            if self.accelerator.is_main_process:            
                # 2.1) De-duplicate the original prompts: 
                # for a batch of size B = (#distinct Q * self.num_generations), we want the distinct Q.
                ordered_unique_prompts = all_prompts_text[:: self.num_generations]
                ordered_unique_ground_truths = all_ground_truths[:: self.num_generations] # Get corresponding GTs

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

                # 2.3) Build the new prompt conditionally based on A1 correctness

                second_turn_prompts = []
                # Iterate over unique prompts, their first answers (A1), and their ground truths (GT)
                for q_text, a1_text, gt_answer in zip(ordered_unique_prompts, first_turn_completions_text, ordered_unique_ground_truths):

                    # Check correctness of the first answer (a1_text) using the provided function
                    # Assuming check_format_and_correctness returns 2 for correct, other values for incorrect.
                    correctness_score = check_correctness(a1_text, gt_answer)
                    second_instruction = "<|im_start|>user\nRate your confidence in the preceding response on a scale of 0 (likely incorrect) to 10 (likely correct), inclusive. Output only your integer confidence score, with the format as such: `<confidence>10</confidence>`.<|im_end|>\n"
                    third_instruction = "<|im_start|>user\nReview your previous response. Considering your confidence score (where 0 indicates likely incorrect and 10 indicates likely correct), solve the problem again. Provide your revised solution in the format: <think> reasoning process here </think> <answer> \\boxed{{final answer inside}} </answer>.<|im_end|>\n"
                    # Determine the second instruction based on correctness
                    if correctness_score == 1:
                        # Correct answer instruction
                        confidence_response = "<|im_start|>assistant\n<confidence>10</confidence><|im_end|>\n"
                    else:
                        # Incorrect answer instruction (handles -2, -1, -0.5)
                        # Note: You wrote "confidence", keeping it as is. If it should be "confident", change below.
                        confidence_response = "<|im_start|>assistant\n<confidence>0</confidence><|im_end|>\n"

                    # Build the second-turn prompt structure: Q + A1 + Conditional Instruction
                    # Ensure the structure matches what your model expects for few-shot/instruction following.
                    # The example structure below follows the pattern from your original code.
                    example = {
                        "prompt": (
                            q_text          # Original prompt
                            + a1_text       # First generated answer
                            + "<|im_end|>\n"  # Optional separators if using chat templates
                            + second_instruction
                            + confidence_response
                            + third_instruction
                            + "<|im_start|>assistant\n<think>"
                        )
                    }
                    # Apply chat template if necessary (as in your original code)
                    processed_prompt = maybe_apply_chat_template(example, self.processing_class)["prompt"]
                    second_turn_prompts.append(processed_prompt)


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
            # print("reward_func:", reward_func)
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
                    # print("REWARD FUNC IS NOT AN NN.MODULE.")
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    a1_texts = [extract_a1_text(text) for text in local_second_turn_prompts]

                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        first_completions=a1_texts,
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
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                # You could optionally do a pretty print here
                # if is_rich_available():
                # print_prompt_completions_sample(
                #     prompts_to_log,
                #     completions_to_log,
                #     rewards_to_log,
                #     self.state.global_step,
                # )
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
    

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                if self.state.global_step % 2 == 0:
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
    
