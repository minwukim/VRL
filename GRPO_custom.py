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

        print("hello")
        print(self.sampling_params)
        print("=========================")
        # 2. First Generation: Generate A1 using a modified sampling parameter (n=1)
        single_sampling_params = deepcopy(self.sampling_params)
        single_sampling_params.n = 1

        if self.args.use_vllm:
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Remove duplicates for faster generation of A1
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                with profiling_context(self, "vLLM.generate (A1)"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=single_sampling_params, use_tqdm=False
                    )
                a1_ids_list = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        a1_ids_list.append(output.token_ids)
            else:
                a1_ids_list = [None] * len(all_prompts_text)
            # MINWU: MAYBE NOT NECESSARY
            # a1_ids_list = broadcast_object_list(a1_ids_list, from_process=0)
            # process_slice = slice(
            #     self.accelerator.process_index * len(prompts),
            #     (self.accelerator.process_index + 1) * len(prompts),
            # )
            # a1_ids_list = a1_ids_list[process_slice]

            # Convert and pad A1 token ids
            a1_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in a1_ids_list]
            a1_ids = pad(a1_ids, padding_value=self.processing_class.pad_token_id)
        else:
            print("SHOULDN'T SEE ME")
            # with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            #     a1_ids = unwrapped_model.generate(
            #         prompt_ids, attention_mask=prompt_mask, generation_config=single_sampling_params
            #     )

        # Decode the generated A1 completions
        a1_text = self.processing_class.batch_decode(a1_ids, skip_special_tokens=True)

        # 3. Construct new prompt: (Q, A1) concatenated with extra context (added_instruction)
        added_instruction = (
            "A conversation between User and Assistant. Given a question and a corresponding response provided below, the Assistant systematically reviews and explains each step of the reasoning process to verify the correctness of the response."
            "If errors are found, the Assistant identifies and corrects them, then re-solves the problem. If the response is correct, the Assistant confirms it and returns the same final answer."
            "The assistant first thinks about the reasoning process in mind, including verification, correction, and resolving the problem if necessary. Then provides the user with the answer."
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>." 
            "The reasoning process, including verification and correction, is enclosed within <think> </think> tags, while the final solution is enclosed within <answer> </answer> tags. The final answer is formatted within \\boxed{{}} to enable direct extraction for grading."
            "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag."
        )

        new_prompts_text = []
        # --- CHANGES MADE BELOW ---
        # Instead of iterating over just the prompt texts and a1_text, we iterate over the full input examples and a1_text.
        # This allows us to extract the 'solution' field and include it in the new prompt.
        for input_example, a1 in zip(inputs, a1_text):
            # Use maybe_apply_chat_template to get the formatted prompt text.
            q_text = maybe_apply_chat_template(input_example, self.processing_class)["prompt"]
            # Extract the solution from the input.
            answer_text = input_example.get("answer", "")
            # Build the new prompt message that includes both the question and the solution.
            example = {
                "prompt": added_instruction + "\n\nQuestion:\n" + extract_math_prompt(q_text) + "\n\nResponse:\n<think> "+ a1 + "\nAssistant: <think>",
                # [
                #     {
                #         "content": added_instruction,
                #         "role": "system"
                #     },
                #     {
                #         "content": "\nQuestion:\n" + remove_tokens(extract_user_text(q_text)) + 
                #                    "\n\nResponse:\n" + remove_tokens(a1) +
                #                    "\n",
                #         "role": "user"
                #     }
                # ],
                "answer": answer_text
            }
            # Replicate for self.sampling_params.n times.
            for i in range(self.sampling_params.n):
                new_prompts_text.append(maybe_apply_chat_template(example, self.processing_class)["prompt"])

        # Preprocess the new prompt (Q, A1, added_instruction)
        new_prompt_inputs = self.processing_class(
            new_prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        new_prompt_inputs = Trainer._prepare_inputs(self, new_prompt_inputs)
        new_prompt_ids, new_prompt_mask = new_prompt_inputs["input_ids"], new_prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            new_prompt_ids = new_prompt_ids[:, -self.max_prompt_length :]
            new_prompt_mask = new_prompt_mask[:, -self.max_prompt_length :]

        # 4. Second Generation: Generate A2 using the original sampling_params (multiple generations)
        if self.args.use_vllm:
            all_new_prompts_text = gather_object(new_prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_new_prompts = list(dict.fromkeys(all_new_prompts_text))
                with profiling_context(self, "vLLM.generate (A2)"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_new_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                a2_ids_list = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        a2_ids_list.append(output.token_ids)
            else:
                a2_ids_list = [None] * len(all_new_prompts_text)
            a2_ids_list = broadcast_object_list(a2_ids_list, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            a2_ids_list = a2_ids_list[process_slice]

            # Convert and pad A2 token ids, then concatenate with the new prompt tokens
            a2_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in a2_ids_list]
            a2_ids = pad(a2_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([new_prompt_ids, a2_ids], dim=1)
        else:
            print("SHOULDN'T SEE ME")
            # MINWU: SHOULD NOT SEE ME
            # with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            #     prompt_completion_ids = unwrapped_model.generate(
            #         new_prompt_ids, attention_mask=new_prompt_mask, generation_config=self.generation_config
            #     )
            # a2_ids = prompt_completion_ids[:, new_prompt_ids.size(1):]

        # 5. Create completion mask: mask tokens after the first EOS in A2
        device = self.accelerator.device
        is_eos = a2_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate the new prompt mask with the A2 completion mask for logit computation
        attention_mask = torch.cat([new_prompt_mask, completion_mask], dim=1)  # (B, P+A2)
        logits_to_keep = a2_ids.size(1)  # we only need logits for A2 tokens

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

        # Decode the generated A2 completions
        completions_text = self.processing_class.batch_decode(a2_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # 6. Compute rewards based on A2 completions, following the original reward processing
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
                        messages = [{"messages": p + c} for p, c in zip(new_prompts_text, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(new_prompts_text, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = Trainer._prepare_inputs(self,reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=new_prompts_text, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # 7. Log metrics as before
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
            prompts_to_log = gather_object(new_prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
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
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # 8. Return final values (using new prompt and A2 completions)
        return {
            "prompt_ids": new_prompt_ids,
            "prompt_mask": new_prompt_mask,
            "completion_ids": a2_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

 



class SwitchingGRPOTrainer(GRPOTrainer):

    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
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

        # 2. First Generation: Generate A1 using a modified sampling parameter (n=1)
        single_sampling_params = deepcopy(self.sampling_params)
        single_sampling_params.n = 1

        if self.args.use_vllm:
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Remove duplicates for faster generation of A1
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                with profiling_context(self, "vLLM.generate (A1)"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=single_sampling_params, use_tqdm=False
                    )
                a1_ids_list = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        a1_ids_list.append(output.token_ids)
            else:
                a1_ids_list = [None] * len(all_prompts_text)
            # MINWU: MAYBE NOT NECESSARY
            # a1_ids_list = broadcast_object_list(a1_ids_list, from_process=0)
            # process_slice = slice(
            #     self.accelerator.process_index * len(prompts),
            #     (self.accelerator.process_index + 1) * len(prompts),
            # )
            # a1_ids_list = a1_ids_list[process_slice]

            # Convert and pad A1 token ids
            a1_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in a1_ids_list]
            a1_ids = pad(a1_ids, padding_value=self.processing_class.pad_token_id)
        else:
            print("SHOULDN'T SEE ME")
            # with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            #     a1_ids = unwrapped_model.generate(
            #         prompt_ids, attention_mask=prompt_mask, generation_config=single_sampling_params
            #     )

        # Decode the generated A1 completions
        a1_text = self.processing_class.batch_decode(a1_ids, skip_special_tokens=True)

        # 3. Construct new prompt: (Q, A1) concatenated with extra context (added_instruction)
        added_instruction = (
            "A conversation between User and Assistant. Given a question and a corresponding response provided below, the Assistant systematically reviews and explains each step of the reasoning process to verify the correctness of the response."
            "If errors are found, the Assistant identifies and corrects them, then re-solves the problem. If the response is correct, the Assistant confirms it and returns the same final answer."
            "The assistant first thinks about the reasoning process in mind, including verification, correction, and resolving the problem if necessary. Then provides the user with the answer."
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>." 
            "The reasoning process, including verification and correction, is enclosed within <think> </think> tags, while the final solution is enclosed within <answer> </answer> tags. The final answer is formatted within \\boxed{{}} to enable direct extraction for grading."
            "User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag."
        )

        new_prompts_text = []
        # --- CHANGES MADE BELOW ---
        # Instead of iterating over just the prompt texts and a1_text, we iterate over the full input examples and a1_text.
        # This allows us to extract the 'solution' field and include it in the new prompt.
        for input_example, a1 in zip(inputs, a1_text):
            # Use maybe_apply_chat_template to get the formatted prompt text.
            q_text = maybe_apply_chat_template(input_example, self.processing_class)["prompt"]
            # Extract the solution from the input.
            answer_text = input_example.get("answer", "")
            # Build the new prompt message that includes both the question and the solution.
            example = {
                "prompt": added_instruction + "\n\nQuestion:\n" + extract_math_prompt(q_text) + "\n\nResponse:\n<think> "+ a1 + "\nAssistant: <think>",
                # [
                #     {
                #         "content": added_instruction,
                #         "role": "system"
                #     },
                #     {
                #         "content": "\nQuestion:\n" + remove_tokens(extract_user_text(q_text)) + 
                #                    "\n\nResponse:\n" + remove_tokens(a1) +
                #                    "\n",
                #         "role": "user"
                #     }
                # ],
                "answer": answer_text
            }
            # Replicate for self.sampling_params.n times.
            for i in range(self.sampling_params.n):
                new_prompts_text.append(maybe_apply_chat_template(example, self.processing_class)["prompt"])

        # Preprocess the new prompt (Q, A1, added_instruction)
        new_prompt_inputs = self.processing_class(
            new_prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        new_prompt_inputs = Trainer._prepare_inputs(self, new_prompt_inputs)
        new_prompt_ids, new_prompt_mask = new_prompt_inputs["input_ids"], new_prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            new_prompt_ids = new_prompt_ids[:, -self.max_prompt_length :]
            new_prompt_mask = new_prompt_mask[:, -self.max_prompt_length :]

        # 4. Second Generation: Generate A2 using the original sampling_params (multiple generations)
        if self.args.use_vllm:
            all_new_prompts_text = gather_object(new_prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_new_prompts = list(dict.fromkeys(all_new_prompts_text))
                with profiling_context(self, "vLLM.generate (A2)"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_new_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                a2_ids_list = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        a2_ids_list.append(output.token_ids)
            else:
                a2_ids_list = [None] * len(all_new_prompts_text)
            a2_ids_list = broadcast_object_list(a2_ids_list, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            a2_ids_list = a2_ids_list[process_slice]

            # Convert and pad A2 token ids, then concatenate with the new prompt tokens
            a2_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in a2_ids_list]
            a2_ids = pad(a2_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([new_prompt_ids, a2_ids], dim=1)
        else:
            print("SHOULDN'T SEE ME")
            # MINWU: SHOULD NOT SEE ME
            # with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            #     prompt_completion_ids = unwrapped_model.generate(
            #         new_prompt_ids, attention_mask=new_prompt_mask, generation_config=self.generation_config
            #     )
            # a2_ids = prompt_completion_ids[:, new_prompt_ids.size(1):]

        # 5. Create completion mask: mask tokens after the first EOS in A2
        device = self.accelerator.device
        is_eos = a2_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate the new prompt mask with the A2 completion mask for logit computation
        attention_mask = torch.cat([new_prompt_mask, completion_mask], dim=1)  # (B, P+A2)
        logits_to_keep = a2_ids.size(1)  # we only need logits for A2 tokens

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

        # Decode the generated A2 completions
        completions_text = self.processing_class.batch_decode(a2_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # 6. Compute rewards based on A2 completions, following the original reward processing
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
                        messages = [{"messages": p + c} for p, c in zip(new_prompts_text, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(new_prompts_text, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = Trainer._prepare_inputs(self,reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=new_prompts_text, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # 7. Log metrics as before
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
            prompts_to_log = gather_object(new_prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
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
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        # 8. Return final values (using new prompt and A2 completions)
        return {
            "prompt_ids": new_prompt_ids,
            "prompt_mask": new_prompt_mask,
            "completion_ids": a2_ids,
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
    

class ReplicatedGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
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
                print("==================")
                print(self.sampling_params)
                print("==================")
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
            print("here")
            # with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            #     prompt_completion_ids = unwrapped_model.generate(
            #         prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
            #     )

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

        # # If all reward functions return None for a given row, issue a detailed warning
        # if torch.isnan(rewards_per_func).all(dim=1).any():
        #     nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
        #     row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
        #     row_reward_kwargs["prompt"] = prompts[nan_row_idx]
        #     row_reward_kwargs["completion"] = completions[nan_row_idx]
        #     warnings.warn(
        #         f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
        #         "Please ensure that at least one reward function returns a valid reward."
        #     )

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
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

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
                # if is_rich_available():
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
