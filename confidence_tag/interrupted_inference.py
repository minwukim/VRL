import re
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class InterruptedInference:
    """
    LLM inference with custom interruption logic to insert <confidence> tags.
    - Segments are detected by '\n\n'
    - A greedy lookahead checks the next word; if it's in a keyword list, we insert a tag
    - After insertion, we REBUILD input_ids so the tag conditions future tokens
    - Early stop on EOS or the literal string '</think>' in the decoded stream
    - Sampling follows Qwen3 'thinking mode' (temperature/top-p/top-k/min-p)
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", top_k_entropy: int = 10):
        print("Loading model and tokenizer...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        assert top_k_entropy > 1, "top_k_entropy must be greater than 1."
        self.top_k_entropy = top_k_entropy
        # theoretical max entropy (ln(k)) for uniform distr over top-k
        self.max_entropy = float(torch.log(torch.tensor(float(self.top_k_entropy))))

        # Lowercased keywords for robust match
        self.keyword_list = {
            'now', 'first', 'second', 'starting', 'suppose',
            'similarly', 'since', 'from', 'given', 'third', 'next',
            'original', 'looking', 'thus', 'therefore', 'ok', 'okay', 'perhaps', 'again',
            'wait', 'alternatively', 'hmm', 'another', 'ah', 'alternative', 'however', 'alright', 'but'
        }

        self.manual_tags_added = 0
        self.self_generated_tags = 0

    # ---------------------------
    # Entropy / Confidence helpers
    # ---------------------------
    def _safe_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.nn.functional.softmax(logits, dim=dim)

    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        Token entropy (natural log) over the top-k tokens ONLY.
        """
        top_k = min(self.top_k_entropy, logits.shape[-1])
        top_k_logits = torch.topk(logits, top_k).values
        probs = self._safe_softmax(top_k_logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -torch.sum(probs * log_probs)
        e = float(entropy.detach().cpu())
        if not (e == e):  # NaN guard
            e = 0.0
        return e

    def _calculate_confidence(self, entropies: List[float]) -> float:
        """
        Confidence = 1 - (avg_entropy / max_entropy), clamped to [0,1].
        """
        if not entropies:
            return 1.0
        avg_entropy = sum(entropies) / len(entropies)
        if not (avg_entropy == avg_entropy) or self.max_entropy <= 0:
            return 1.0
        conf = 1.0 - (avg_entropy / self.max_entropy)
        return max(0.0, min(1.0, conf))

    # ---------------------------
    # Decoding helpers
    # ---------------------------
    def _decode_ids(self, ids: List[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _first_double_newline_token_index(self, ids: List[int]) -> int:
        """
        Return the smallest token index i (1-based slice end) such that decode(ids[:i]) contains '\n\n'.
        If none, return -1.
        """
        for i in range(1, len(ids) + 1):
            s = self._decode_ids(ids[:i])
            if "\n\n" in s:
                return i
        return -1

    # ---------------------------
    # Qwen3 "thinking mode" sampler
    # ---------------------------
    def _sample_next_token(
        self,
        logits: torch.Tensor,            # shape [vocab]
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
    ) -> int:
        """
        Qwen3 thinking-mode sampler:
          - temperature (default 0.6)
          - top-k (default 20)
          - top-p (default 0.95)
          - min-p (default 0.0 → disabled)
        Avoid greedy decoding in thinking mode.
        """
        # Temperature
        if temperature is None or temperature <= 0.0:
            # per model card: avoid greedy; fallback to 0.6
            temperature = 0.6

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # Top-k
        if top_k is not None and top_k > 0:
            k = min(top_k, probs.numel())
            topk_vals, topk_idx = torch.topk(probs, k=k)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[topk_idx] = True
            probs = torch.where(mask, probs, torch.zeros_like(probs))

        # Top-p (nucleus)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum <= top_p
            if not cutoff.any():
                cutoff[0] = True  # keep the top-1 at least
            keep_idx = sorted_idx[cutoff]
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[keep_idx] = True
            probs = torch.where(mask, probs, torch.zeros_like(probs))

        # Min-p (rarely used; leave default 0.0)
        if min_p is not None and min_p > 0.0:
            probs = torch.where(probs >= min_p, probs, torch.zeros_like(probs))

        # Renormalize
        s = probs.sum()
        if s.item() == 0.0:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = probs / s

        next_id = torch.multinomial(probs, num_samples=1).item()
        return int(next_id)

    # ---------------------------
    # Main generation
    # ---------------------------
    def generate_with_confidence(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
    ) -> str:
        """
        Token-by-token generation; insert <confidence> tags after segments ending with '\n\n'
        when lookahead suggests a new step/keyword. After insertion, rebuild input_ids so
        the tag conditions future tokens. Early-stop on EOS or literal '</think>'.
        """
        self.manual_tags_added = 0
        self.self_generated_tags = 0

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_ids = prompt_ids.clone()

        generated_ids: List[int] = []
        current_segment_ids: List[int] = []
        current_segment_entropies: List[float] = []

        full_generated_text = ""
        decoded_so_far = ""

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab]

            # Qwen3 thinking-mode sampling
            next_token = self._sample_next_token(
                logits=next_token_logits.squeeze(0),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )
            next_token_id = torch.tensor([[next_token]], device=input_ids.device)

            # EOS stop
            if next_token == self.tokenizer.eos_token_id:
                break

            # Update states
            generated_ids.append(next_token)
            current_segment_ids.append(next_token)
            current_segment_entropies.append(self._calculate_entropy(next_token_logits.squeeze(0)))

            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # String-level early stop
            decoded_piece = self._decode_ids([next_token])
            decoded_so_far += decoded_piece
            if "</think>" in decoded_so_far:
                break

            # Segmenting by '\n\n'
            current_segment_text = self._decode_ids(current_segment_ids)
            if "\n\n" in current_segment_text:
                split_idx = self._first_double_newline_token_index(current_segment_ids)
                if split_idx == -1:
                    continue

                ids_for_conf = current_segment_ids[:split_idx]
                ents_for_conf = current_segment_entropies[:split_idx]
                remaining_ids = current_segment_ids[split_idx:]

                # Greedy lookahead to next word (keep greedy for stability)
                lookahead_input_ids = input_ids.clone()
                lookahead_word_ids: List[int] = []
                lookahead_word_str = ""
                for __ in range(10):
                    with torch.no_grad():
                        la_out = self.model(lookahead_input_ids)
                    la_logits = la_out.logits[:, -1, :]
                    la_next = torch.argmax(la_logits, dim=-1).unsqueeze(0)
                    la_int = int(la_next.item())
                    if la_int == self.tokenizer.eos_token_id:
                        break
                    lookahead_input_ids = torch.cat([lookahead_input_ids, la_next], dim=-1)
                    lookahead_word_ids.append(la_int)
                    token_str = self.tokenizer.decode(la_next[0], skip_special_tokens=True)
                    lookahead_word_str += token_str
                    if " " in lookahead_word_str:
                        break

                cleaned_word = re.sub(r'[^a-zA-Z]', '', lookahead_word_str.split(" ")[0]).lower()

                seg_text = self._decode_ids(ids_for_conf)
                if not seg_text.strip():
                    current_segment_ids = remaining_ids
                    current_segment_entropies = current_segment_entropies[split_idx:]
                    continue

                if cleaned_word in self.keyword_list:
                    # avoid double insertion if the model self-generated a tag
                    if re.search(r'</confidence>\s*?$', seg_text.rstrip('\n')):
                        full_generated_text += seg_text
                        self.self_generated_tags += 1

                        # Rewind to split and rebuild state (no new tag)
                        N_before_segment = len(generated_ids) - len(current_segment_ids)
                        split_global_index = N_before_segment + split_idx
                        generated_ids = generated_ids[:split_global_index]
                        input_ids = torch.cat([prompt_ids, torch.tensor([generated_ids], device=self.device)], dim=-1)

                        current_segment_ids = []
                        current_segment_entropies = []
                        continue
                    else:
                        # Insert tag
                        confidence = self._calculate_confidence(ents_for_conf)
                        confidence_tag = f" <confidence> {confidence:.2f} </confidence>\n\n"
                        full_generated_text += seg_text.rstrip('\n') + confidence_tag
                        self.manual_tags_added += 1

                        # Make the tag affect future tokens
                        N_before_segment = len(generated_ids) - len(current_segment_ids)
                        split_global_index = N_before_segment + split_idx
                        generated_ids = generated_ids[:split_global_index]
                        tag_ids = self.tokenizer.encode(confidence_tag, add_special_tokens=False)
                        generated_ids.extend(tag_ids)
                        input_ids = torch.cat([prompt_ids, torch.tensor([generated_ids], device=self.device)], dim=-1)

                        # Reset segment
                        current_segment_ids = []
                        current_segment_entropies = []
                else:
                    # No tag; commit up to boundary and continue
                    full_generated_text += seg_text
                    current_segment_ids = remaining_ids
                    current_segment_entropies = current_segment_entropies[split_idx:]

        # Flush tail
        if current_segment_ids:
            full_generated_text += self._decode_ids(current_segment_ids)

        return full_generated_text.strip()
