# interrupted_inference_vllm.py

import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _clean_first_word(s: str) -> str:
    first = s.strip().split(" ")[0] if s.strip() else ""
    return re.sub(r"[^a-zA-Z]", "", first).lower()


def _entropy_from_logprobs(step_logprobs: Optional[dict], top_k: int) -> float:
    """
    Compute entropy using up to the top-k logprobs returned by vLLM for a single step.
    step_logprobs: dict[token -> LogProb] where value has .logprob
    """
    if not step_logprobs:
        return 0.0
    vals = []
    for i, lp in enumerate(step_logprobs.values()):
        if i >= top_k:
            break
        vals.append(lp.logprob)
    if not vals:
        return 0.0
    log_probs = torch.tensor(vals, dtype=torch.float32)
    probs = torch.exp(log_probs)
    s = probs.sum()
    if s.item() <= 0:
        return 0.0
    probs = probs / s
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    return float(entropy)


class InterruptedInferenceVLLM:
    """
    vLLM-based generator that:
      - generates chunk-by-chunk with stop strings: ["\\n\\n", "</think>", "<confidence>"]
      - computes confidence from per-token entropies (top-k of vLLM logprobs) for each segment
      - on stop "<confidence>", immediately inserts a computed <confidence> tag and continues
      - on stop "\\n\\n", optionally inserts a tag depending on lookahead keyword
      - on stop "</think>", stops
      - uses prefix caching for speed
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        top_k_entropy: int = 10,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        gpu_mem_util: float = 0.90,
    ):
        self.top_k_entropy = max(2, int(top_k_entropy))
        self.max_entropy = float(torch.log(torch.tensor(float(self.top_k_entropy))))

        self.keyword_list = {
            'now', 'first', 'second', 'starting', 'suppose',
            'similarly', 'since', 'from', 'given', 'third', 'next',
            'original', 'looking', 'thus', 'therefore', 'ok', 'okay', 'perhaps', 'again',
            'wait', 'alternatively', 'hmm', 'another', 'ah', 'alternative', 'however', 'alright', 'but'
        }

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.manual_tags_added = 0
        self.self_generated_tags = 0

        self.STOP_NN = "\n\n"
        self.STOP_THINK_END = "</think>"
        self.STOP_CONF = "<confidence>"
        self.STOPS = [self.STOP_NN, self.STOP_THINK_END, self.STOP_CONF]

    # ---------- helpers ----------
    def _confidence_from_entropies(self, entropies: List[float]) -> float:
        if not entropies or self.max_entropy <= 0:
            return 1.0
        avg_entropy = sum(entropies) / len(entropies)
        conf = 1.0 - (avg_entropy / self.max_entropy)
        return max(0.0, min(1.0, conf))

    def _generate_once(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        logprobs_k: int,
        stops: List[str],
        seed: Optional[int] = None,
    ) -> Tuple[str, List[dict], Optional[str]]:
        """
        Single vLLM generate with logprobs and stop strings.
        Returns (generated_text, per_step_logprobs_list, stop_reason_str_or_None).
        """
        # vLLM requires top_k == -1 (disable) or >= 1
        if top_k is None or top_k <= 0:
            top_k = -1

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=1,
            stop=stops,
            logprobs=logprobs_k,
            seed=seed,
        )
        out = self.llm.generate([prompt], sp)[0]
        seq = out.outputs[0]

        stop_reason = getattr(seq, "stop_reason", None)
        return seq.text, (seq.logprobs or []), stop_reason

    def _greedy_peek(self, prompt: str, max_tokens: int = 8) -> str:
        """
        Greedy continuation to inspect what's next from 'prompt'.
        Use top_k=-1 (disabled) to satisfy vLLM constraints.
        """
        txt, _, _ = self._generate_once(
            prompt=prompt,
            temperature=0.0,   # greedy
            top_p=1.0,
            top_k=-1,         # disable top-k properly
            max_tokens=max_tokens,
            logprobs_k=0,
            stops=[],         # no stop; raw peek
            seed=0,           # seed irrelevant for greedy
        )
        return txt

    # ---------- main ----------
    def generate_with_confidence(
        self,
        prompt: str,
        total_max_new_tokens: int = 4096,
        step_max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        seed: Optional[int] = None,
        **kwargs,  # tolerate unexpected kwargs for compatibility
    ) -> str:
        """
        Chunked generation loop with stop strings:
          - generate until one of ["\\n\\n", "</think>", "<confidence>"]
          - compute confidence from the per-token entropies of the chunk
          - If stop == "</think>": append chunk and stop
          - If stop == "<confidence>": append chunk + computed tag immediately
          - If stop == "\\n\\n": do keyword lookahead to decide inserting tag
          - Continue with updated context (prefix cache makes this fast)
        """
        self.manual_tags_added = 0
        self.self_generated_tags = 0

        context = prompt
        rendered = ""  # accumulated visible text after the prompt
        tokens_used = 0

        # prefer explicit seed; otherwise accept kw fallback if provided
        base_seed = seed if seed is not None else kwargs.get("seed", None)

        while tokens_used < total_max_new_tokens:
            # 1) Generate a chunk up to next stop
            chunk_text, step_logprobs, stop_reason = self._generate_once(
                prompt=context,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=min(step_max_tokens, total_max_new_tokens - tokens_used),
                logprobs_k=self.top_k_entropy,
                stops=self.STOPS,
                seed=base_seed,
            )

            if not chunk_text and stop_reason is None:
                break

            # 2) Compute confidence for this chunk
            entropies = [_entropy_from_logprobs(lp, self.top_k_entropy) for lp in step_logprobs]
            confidence = self._confidence_from_entropies(entropies)

            # 3) Track token budget
            new_tokens = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
            tokens_used += new_tokens

            # 4) Determine which stop triggered
            detected_stop = stop_reason
            if detected_stop is None:
                peek = self._greedy_peek(context + chunk_text, max_tokens=8)
                if peek.startswith(self.STOP_THINK_END):
                    detected_stop = self.STOP_THINK_END
                elif peek.startswith(self.STOP_CONF):
                    detected_stop = self.STOP_CONF
                elif peek.startswith(self.STOP_NN):
                    detected_stop = self.STOP_NN
                else:
                    detected_stop = self.STOP_NN  # safe default

            # 5) Handle according to stop
            if detected_stop == self.STOP_THINK_END:
                rendered += chunk_text
                break

            elif detected_stop == self.STOP_CONF:
                tag = f" <confidence> {confidence:.2f} </confidence>\n\n"
                rendered += chunk_text.rstrip('\n') + tag
                # (bonus not applied) self.self_generated_tags is intentionally not incremented
                self.manual_tags_added += 1
                context = prompt + rendered

            elif detected_stop == self.STOP_NN:
                lookahead_prompt = context + chunk_text + self.STOP_NN
                first_word_snippet = self._greedy_peek(lookahead_prompt, max_tokens=8)
                cleaned_word = _clean_first_word(first_word_snippet)

                if cleaned_word in self.keyword_list:
                    tag = f" <confidence> {confidence:.2f} </confidence>\n\n"
                    rendered += chunk_text.rstrip('\n') + tag
                    self.manual_tags_added += 1
                else:
                    rendered += chunk_text + self.STOP_NN
                context = prompt + rendered

            else:
                rendered += chunk_text + self.STOP_NN
                context = prompt + rendered

            if tokens_used >= total_max_new_tokens:
                break

        return rendered.strip()
