"""Stateless GPU Qwen interface for RASR's experimental dynamic word search."""

import re


def decode_byte_pieces(text):
    """Decode contiguous SentencePiece byte runs, retaining normal characters.

    Invalid UTF-8 is replaced, as in SentencePiece decoding, not silently dropped.
    This conversion precedes casing and LLM tokenization and affects no AM ids.
    """
    return re.sub(
        r"(?:<0x[0-9A-Fa-f]{2}>)+",
        lambda m: bytes(int(x, 16) for x in re.findall(r"<0x([0-9A-Fa-f]{2})>", m[0])).decode("utf-8", "replace"),
        text,
    )


def casing_surfaces(text, casings):
    text = decode_byte_pieces(text)
    word = text.lstrip()
    separator = text[: len(text) - len(word)]
    functions = dict(
        original=lambda w: w, lower=str.lower, capitalized=lambda w: w[:1].upper() + w[1:].lower(), upper=str.upper
    )
    return list(dict.fromkeys(separator + functions[c](word) for c in casings))


class QwenContinuationScorer:
    """Pure Python implementation, wrapped in the selected librasr at runtime."""

    def __init__(self, repo, *, device="cuda", casings=("original", "lower", "capitalized", "upper"), batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.casings, self.batch_size = device, tuple(casings), batch_size
        if batch_size < 1 or not self.casings or set(self.casings) - {"original", "lower", "capitalized", "upper"}:
            raise ValueError("Invalid batch size/casings")
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(device).eval()
        bos, eos = self.model.config.bos_token_id, self.model.config.eos_token_id
        bos = bos if bos is not None else self.tokenizer.eos_token_id
        if not isinstance(bos, int) or not isinstance(eos, int):
            raise ValueError("Expected singleton Qwen BOS/EOS")
        self.initial, self.end = [bos], [eos]
        self.reset()

    def reset(self):
        self.requests = self.forward_batches = self.max_request_length = 0

    def initial_tokens(self):
        return self.initial

    def sentence_end_tokens(self):
        return self.end

    def tokenize(self, texts):
        result = []
        for text in texts:
            variants = []
            for surface in casing_surfaces(text, self.casings):
                ids = self.tokenizer.encode(surface, add_special_tokens=False)
                if not ids or set(ids) & set(self.tokenizer.all_special_ids):
                    raise ValueError(f"Empty/special tokenization of {surface!r}")
                if self.tokenizer.decode(ids, clean_up_tokenization_spaces=False) != surface:
                    raise ValueError(f"Non-round-tripping tokenization of {surface!r}")
                if ids not in variants:
                    variants.append(ids)
            result.append(variants)
        return result

    def score_continuations(self, prefixes, continuations):
        import torch

        if len(prefixes) != len(continuations) or any(not p for p in prefixes):
            raise ValueError("Mismatched requests or empty prefix")
        result = [[] for _ in prefixes]
        order = sorted(
            (i for i, c in enumerate(continuations) if c), key=lambda i: len(prefixes[i]) + len(continuations[i])
        )
        self.requests += len(order)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                length = max(len(prefixes[i]) + len(continuations[i]) for i in batch)
                if length > self.model.config.max_position_embeddings:
                    raise ValueError("LLM context limit exceeded; no truncation")
                self.max_request_length = max(self.max_request_length, length)
                self.forward_batches += 1
                ids = torch.full((len(batch), length), self.end[0], dtype=torch.long, device=self.device)
                mask = torch.zeros_like(ids)
                for b, i in enumerate(batch):
                    seq = prefixes[i] + continuations[i]
                    ids[b, : len(seq)] = torch.tensor(seq, device=self.device)
                    mask[b, : len(seq)] = 1
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                # Only continuation positions need a softmax; avoid materializing
                # float32 log probabilities for every token of every full history.
                for b, i in enumerate(batch):
                    p, c = prefixes[i], continuations[i]
                    selected = logits[b, len(p) - 1 : len(p) + len(c) - 1].float()
                    costs = torch.nn.functional.cross_entropy(
                        selected, torch.tensor(c, device=self.device), reduction="none"
                    )
                    if not torch.isfinite(costs).all():
                        raise ValueError("Non-finite Qwen token costs")
                    result[i] = costs.cpu().tolist()
        return result


def register_qwen_scorer(librasr, *, zero=False, instances=None):
    """Register only against the explicitly selected binary, not at import time."""

    class Scorer(librasr.LlmScorer):
        def __init__(self, config):
            super().__init__(config)
            self.backend = (
                None
                if zero
                else QwenContinuationScorer(
                    config["model"],
                    device="cuda",
                    batch_size=int(config["max-batch-size"] or 8),
                    casings=tuple((config["case-variants"] or "original,lower,capitalized,upper").split(",")),
                )
            )
            if instances is not None:
                instances.append(self)

        def reset(self):
            if self.backend:
                self.backend.reset()

        def initial_tokens(self):
            return [0] if zero else self.backend.initial_tokens()

        def sentence_end_tokens(self):
            return [] if zero else self.backend.sentence_end_tokens()

        def tokenize(self, texts):
            return [[[1]] for _ in texts] if zero else self.backend.tokenize(texts)

        def score_continuations(self, prefixes, continuations):
            return (
                [[0.0] * len(c) for c in continuations]
                if zero
                else self.backend.score_continuations(prefixes, continuations)
            )

    librasr.register_llm_scorer_type("qwen-dynamic-zero" if zero else "qwen-dynamic", Scorer)
