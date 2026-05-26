from typing import Optional, Union, Any, Sequence, List, Dict
import time
import numpy as np
import torch
from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats
from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from ..logits_transform import make_logits_transform
from .base import BaseModelInterface, ForwardOutput


def _unwrap_checkpoint_wrappers(model) -> int:
    """Walk `model` and replace any `CheckpointWrapper` (torch.distributed) with the
    wrapped module directly. Returns the number unwrapped.

    Phi4-MM's audio encoder config enables `activation_checkpointing` which wraps
    layers with REENTRANT-impl checkpointing -- incompatible with
    `torch.autograd.grad()` on torch>=2.7. Inference-only use here, so we just
    skip checkpointing entirely (small memory bump, no semantic change).
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    count = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, CheckpointWrapper):
                setattr(module, name, child._checkpoint_wrapped_module)
                count += 1
    return count


class Phi4MM(BaseModelInterface):
    """
    Phi4MM model interface.

    Beyond the basic transcription-loss machinery, supports several optional
    variant flags useful for grad-based alignment experiments. All flags
    default to behavior-preserving values; the base behavior is unchanged
    unless a flag is explicitly enabled.

    Per-word loss-formulation variants (gated):
    - ``fake_loss_grad``: inject ``1/V - one_hot(target)`` at logits
      (independent of model confidence). Replicates an older trick.
    - ``margin_grad`` (with ``margin_grad_k``): contrastive
      ``log p[target] - logsumexp(top-K non-target competitors)``.
      K=1 is the simple top-1 margin. Mutually exclusive with ``fake_loss_grad``.
    - ``eos_margin``: ``log p[target] - log p[<|end|>]``: margin against
      "stop now" rather than predicting the target.
    - ``first_subword_only``: per word, only the first subword's log_p
      contributes to the per-word loss (ignore continuation subwords).

    Reference-tokenization variants (gated, char-level only):
    - ``char_level``: tokenize the reference at character level (each char
      becomes a "word" from Phi4MM's perspective). Per-word log_p becomes a
      sum over per-character log_p.
    - ``char_level_sep`` (e.g. ``" "`` or ``"·"``): inserted as a
      separator char between consecutive words in char_level mode -- gives
      the model an explicit inter-word boundary cue.
    - ``char_level_brackets`` in ``("char", "word")``: wrap each word with
      ``[`` / ``]`` (when ``"char"``) or ``[BOW]`` / ``[EOW]`` (when ``"word"``)
      tokens. Only effective with ``char_level=True``.

    Torch 2.7 compat (gated):
    - ``unwrap_checkpoint_wrappers``: walk the model and replace any
      ``CheckpointWrapper`` (REENTRANT-impl, incompatible with
      ``torch.autograd.grad()``) with the wrapped module directly.
    - ``target_start_end_to_device``: move ``target_start_end`` from CPU
      to ``self.device`` before returning from forward(), so downstream
      device-mixed ops don't complain.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        speech_prompt: str = "Transcribe the audio clip into text.",
        grad_wrt: str = "speech_embeddings",
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
        # Variant flags. Defaults preserve original behavior.
        fake_loss_grad: bool = False,
        margin_grad: bool = False,
        margin_grad_k: int = 1,
        eos_margin: bool = False,
        first_subword_only: bool = False,
        char_level: bool = False,
        char_level_sep: Optional[str] = None,
        char_level_brackets: Optional[str] = None,
        unwrap_checkpoint_wrappers: bool = False,
        target_start_end_to_device: bool = False,
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param speech_prompt: text-only part of the prompt
        """
        super().__init__()

        if fake_loss_grad and margin_grad:
            raise ValueError("fake_loss_grad and margin_grad are mutually exclusive")
        if margin_grad_k < 1:
            raise ValueError(f"margin_grad_k must be >= 1, got {margin_grad_k}")
        if char_level_brackets not in (None, "char", "word"):
            raise ValueError(f"char_level_brackets must be None / 'char' / 'word', got {char_level_brackets!r}")

        self.device = device
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.grad_wrt = grad_wrt
        self.logits_transform = make_logits_transform(logits_transform)

        self._fake_loss_grad = fake_loss_grad
        self._margin_grad = margin_grad
        self._margin_grad_k = margin_grad_k
        self._eos_margin = eos_margin
        self._first_subword_only = first_subword_only
        self._char_level = char_level
        self._char_level_sep = char_level_sep
        self._char_level_brackets = char_level_brackets
        self._target_start_end_to_device = target_start_end_to_device

        print("Import Transformers...")
        start_time = time.time()

        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"({time.time() - start_time} secs)")

        print("Loading model...")
        start_time = time.time()
        model_dir = get_content_dir_from_hub_cache_dir(self.model_dir)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype="auto", trust_remote_code=True, device_map=str(device)
        ).to(device)

        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalForCausalLM

        model: Phi4MultimodalForCausalLM  # just as an example...
        print(model)
        print("model.dtype:", model.dtype)
        print(f"({time.time() - start_time} secs)")

        self.processor = processor
        self.model = model

        tokenizer = self.processor.tokenizer
        (self.assistant_token_id,) = tokenizer.convert_tokens_to_ids(["<|assistant|>"])
        (self.assistant_end_token_id,) = tokenizer.convert_tokens_to_ids(["<|end|>"])

        if unwrap_checkpoint_wrappers:
            n = _unwrap_checkpoint_wrappers(self.model)
            print(f"Phi4MM: unwrapped {n} CheckpointWrapper(s)")

        print(
            f"Phi4MM variant flags: fake_loss_grad={fake_loss_grad} margin_grad={margin_grad}"
            f" margin_grad_k={margin_grad_k} eos_margin={eos_margin}"
            f" first_subword_only={first_subword_only} char_level={char_level}"
            f" char_level_sep={char_level_sep!r} char_level_brackets={char_level_brackets!r}"
        )

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        # char_level pre-processing: explode each word into per-char "words"
        # (optionally with separators and BOW/EOW wrappers), call the core
        # forward, then re-group target_start_end back to one entry per word.
        if self._char_level and raw_targets is not None:
            assert len(raw_targets) == 1, "char_level supports batch size 1 only"
            orig_words = raw_targets[0]
            sep = self._char_level_sep
            brk = self._char_level_brackets
            bow_tok = "[BOW]" if brk == "word" else ("[" if brk == "char" else None)
            eow_tok = "[EOW]" if brk == "word" else ("]" if brk == "char" else None)
            chars: list = []
            word_char_ranges: list = []
            for i, word in enumerate(orig_words):
                if i > 0 and sep is not None:
                    chars.append(sep)
                cstart = len(chars)
                if bow_tok is not None:
                    chars.append(bow_tok)
                chars.extend(word)
                if eow_tok is not None:
                    chars.append(eow_tok)
                word_char_ranges.append((cstart, len(chars)))
            core_raw_targets = [chars]
            core_raw_target_seq_lens = torch.tensor([len(chars)])
        else:
            orig_words = None
            word_char_ranges = None
            core_raw_targets = raw_targets
            core_raw_target_seq_lens = raw_target_seq_lens

        out = self._forward_core(
            raw_inputs=raw_inputs,
            raw_inputs_sample_rate=raw_inputs_sample_rate,
            raw_input_seq_lens=raw_input_seq_lens,
            raw_targets=core_raw_targets,
            raw_target_seq_lens=core_raw_target_seq_lens,
            omitted_prev_context=omitted_prev_context,
        )

        if word_char_ranges is not None:
            tse = out.target_start_end  # [B, num_chars+1, 2] (chars + EOS)
            assert tse.shape[1] == len(core_raw_targets[0]) + 1, (
                f"char_level: expected {len(core_raw_targets[0]) + 1} entries (chars + EOS), got {tse.shape[1]}"
            )
            new_entries = []
            for cstart, cend in word_char_ranges:
                t0 = int(tse[0, cstart, 0])
                t1 = int(tse[0, cend - 1, 1])
                new_entries.append([t0, t1])
            new_entries.append([int(tse[0, -1, 0]), int(tse[0, -1, 1])])
            out.target_start_end = torch.tensor([new_entries], dtype=tse.dtype, device=tse.device)
            out.target_seq_lens = torch.tensor([len(orig_words)])

        if self._target_start_end_to_device and out.target_start_end.device != self.device:
            out.target_start_end = out.target_start_end.to(self.device)
        if self._first_subword_only:
            tse = out.target_start_end  # [B, num_words+1, 2]
            out.target_start_end = torch.stack([tse[..., 0], tse[..., 0] + 1], dim=-1)
        return out

    def _forward_core(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        # Original forward() body. Unchanged so default behavior (no variant
        # flags set) is byte-identical to the pre-refactor Phi4MM.
        assert raw_inputs_sample_rate is not None  # assume audio input
        assert (len(raw_inputs),) == raw_input_seq_lens.shape == (len(raw_targets),) == raw_target_seq_lens.shape, (
            f"batch size inconsistent, got {len(raw_inputs)=}, {len(raw_input_seq_lens)=},"
            f" {len(raw_targets)=}, {len(raw_target_seq_lens)=}"
        )
        assert len(raw_inputs) == 1, f"Phi4MM currently supports only batch size 1, got {len(raw_inputs)=}"
        assert isinstance(raw_inputs, torch.Tensor), f"raw_inputs expected to be torch.Tensor, got {type(raw_inputs)=}"
        assert raw_inputs.ndim == 2, f"raw_inputs expected to be 2D, got {raw_inputs.ndim=}"
        assert raw_input_seq_lens[0] == raw_inputs.shape[1], (
            f"raw_input_seq_lens[0]={raw_input_seq_lens[0]} != {raw_inputs.shape[1]}"
        )

        speech_prompt = self.speech_prompt
        dev = self.device

        # TODO maybe monkey patch some modules, e.g. Phi4MMRMSNorm,
        #   via liger_kernel.transformers.monkey_patch._patch_rms_norm_module?

        tokenizer = self.processor.tokenizer

        words = raw_targets[0]
        transcription = " ".join(words)
        added_prefix = False
        if omitted_prev_context is not None and omitted_prev_context[0] > 0:
            added_prefix = True
            transcription = "... " + transcription
        prompt = f"<|user|><|audio_1|>{speech_prompt}<|end|><|assistant|>{transcription}<|end|>"
        inputs = self.processor(text=prompt, audios=[(raw_inputs[0], raw_inputs_sample_rate)], return_tensors="pt")
        input_ids = inputs["input_ids"]
        (dst_text_start,) = torch.nonzero(input_ids[0] == self.assistant_token_id).squeeze(dim=1)
        dst_text_start = int(dst_text_start) + 1  # one past the assistant token
        dst_text_end = input_ids.shape[-1] - 1  # right before the <end> token. excluding.
        inputs = inputs.to(dev)
        input_ids = inputs["input_ids"]
        inputs_embeds = inputs["input_audio_embeds"]
        print("inputs_embeds:", inputs_embeds)
        inputs_embeds.requires_grad = True
        inputs_embeds.retain_grad()
        # We don't need the logits here. There is currently no way to not compute them,
        # so num_logits_to_keep=1 is the best we can do.
        # We then will compute only the needed logits below,
        # and for that, we need the last layer output, thus output_hidden_states=True.
        res = self.model(**inputs, output_hidden_states=True, num_logits_to_keep=1)
        last_out = res.hidden_states[-1]  # [B,T,D]
        del res
        assert last_out.shape[:2] == input_ids.shape
        report_dev_memory_stats(dev)

        targets = input_ids[:, dst_text_start:dst_text_end]  # [B,T']
        n_targets = targets.shape[1]
        # Append the assistant-end token (EOS) to targets so the chunk-exit log_prob
        # can be looked up by batches_gather. target_start_end gets a matching entry
        # below (one past the last word).
        targets = torch.cat(
            [targets, torch.tensor([[self.assistant_end_token_id]], device=targets.device, dtype=targets.dtype)],
            dim=1,
        )  # [B, T'+1]

        words_start_end = [[0, 1]]
        tokens = [tokenizer.decode(targets[0, :1])]
        words_ = [tokens[-1]]
        for t in range(1, n_targets):
            s = tokenizer.decode(targets[0, t : t + 1])
            tokens.append(s)
            if s.startswith(" "):  # new word
                words_.append(s[1:])
                words_start_end[-1][1] = t
                words_start_end.append([t, t + 1])
            else:
                words_[-1] += s
                words_start_end[-1][1] = t + 1
        if added_prefix:
            assert words_[0] == "..."
            words_start_end = words_start_end[1:]
            words_ = words_[1:]
        assert len(words_start_end) == len(words_) == len(words), f"got {tokens=}"
        assert words_ == words, f"{tokens=} {words=} {words_=}"
        # Add a matching entry for the EOS token appended above (one past the last word).
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]

        assert self.grad_wrt == "speech_embeddings", f"{self.grad_wrt=!r}"

        # Map each model-internal speech-embedding frame back to its raw-audio sample
        # range. Phi4-MM's audio front-end uses log-mel features at a 10ms hop (160
        # samples @ 16kHz). For arbitrary sample rates we just linearly interpolate
        # (raw samples are evenly divided across frames). This matches the timestamp
        # convention used in :mod:`exp2025_05_05_align` (linear-interp frame -> time).
        n_frames = inputs_embeds.shape[1]
        n_samples = int(raw_input_seq_lens[0])  # B=1 enforced above
        edges = torch.arange(n_frames + 1, dtype=torch.float64) * (n_samples / n_frames)
        input_raw_start_end = torch.stack(
            [edges[:-1].round().long(), edges[1:].round().long()], dim=-1
        ).unsqueeze(0)  # [B=1, n_frames, 2]

        return ForwardOutput(
            inputs=inputs_embeds,
            input_seq_lens=torch.tensor([inputs_embeds.shape[1]]),  # [B]
            input_slice_start_end=None,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),  # [B]
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64).unsqueeze(0),  # [B, T, 2]
            outputs=dict(dst_text_start=dst_text_start, last_out=last_out),
        )

    def log_probs(
        self, *, forward_output: ForwardOutput, start: Union[int, torch.Tensor], end: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        if self._fake_loss_grad:
            log_p = self._log_probs_fake_loss(forward_output=forward_output, start=start, end=end)
        else:
            log_p = self._log_probs_core(forward_output=forward_output, start=start, end=end)

        if self._margin_grad:
            log_p = self._apply_margin(log_p, k=self._margin_grad_k)
        if self._eos_margin:
            # Subtract log_p[<|end|>] from every position. After batches_gather picks
            # position v = target, we get log_p[target] - log_p[end]: margin against
            # "stop now" rather than predicting the target.
            eos_idx = self.assistant_end_token_id
            log_p = log_p - log_p[..., eos_idx : eos_idx + 1]
        return log_p

    def _log_probs_core(
        self, *, forward_output: ForwardOutput, start: Union[int, torch.Tensor], end: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        # Original log_probs() body. Unchanged so default behavior (no variant
        # flags set) is byte-identical to the pre-refactor Phi4MM.
        last_out = forward_output.outputs["last_out"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        last_out = batch_slice(last_out, (dst_text_start + start - 1, dst_text_start + end - 1))

        logits = self.model.lm_head(last_out)  # [B, T', V]
        logits = logits.float()
        for f in self.logits_transform:
            logits = f(logits)

        return logits.log_softmax(-1)  # [B, T', V]

    def _log_probs_fake_loss(
        self, *, forward_output: ForwardOutput, start: Union[int, torch.Tensor], end: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        # Forward value of fake_logits is 0 (so log_softmax = log(1/V) for every class),
        # but gradient flows through real `logits`. Replicates the OLD
        # exp2025_05_05_align fake-loss-grad trick.
        last_out = forward_output.outputs["last_out"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        last_out = batch_slice(last_out, (dst_text_start + start - 1, dst_text_start + end - 1))
        logits = self.model.lm_head(last_out)
        logits = logits.float()
        for f in self.logits_transform:
            logits = f(logits)
        fake_logits = logits + (-logits).detach()  # value: 0; grads -> real logits
        return fake_logits.log_softmax(-1)

    def _apply_margin(self, log_p: torch.Tensor, *, k: int) -> torch.Tensor:
        """Subtract `logsumexp(top-K of log_p, excluding v)` from each (B, T', v)
        position. When batches_gather later picks v = target, the result is
        ``log_p[target] - logsumexp(top-K non-target competitors)``. K=1 reduces
        to the simple top-1 case (separate codepath, kept byte-exact for hash safety).
        """
        V = log_p.shape[-1]
        if k == 1:
            top2_vals, top2_idx = log_p.topk(2, dim=-1)  # values, indices: [B, T', 2]
            top1_val = top2_vals[..., 0:1]
            top2_val = top2_vals[..., 1:2]
            top1_idx = top2_idx[..., 0:1]
            arange_v = torch.arange(V, device=log_p.device)
            is_top1 = arange_v.view(1, 1, V) == top1_idx  # [B, T', V]
            top_non_i = torch.where(is_top1, top2_val.expand_as(log_p), top1_val.expand_as(log_p))
            return log_p - top_non_i
        topK1_vals, topK1_idx = log_p.topk(k + 1, dim=-1)  # [B, T', K+1]
        arange_v = torch.arange(V, device=log_p.device)
        is_in_topK1 = (topK1_idx.unsqueeze(-1) == arange_v.view(1, 1, 1, V)).any(dim=-2)  # [B, T', V]
        # For v not in top-(K+1): subtract logsumexp(top-K).
        lse_topK = topK1_vals[..., :k].logsumexp(dim=-1, keepdim=True)  # [B, T', 1]
        # For v in top-(K+1): subtract log(exp(lse_topK+1) - exp(log_p[v])).
        lse_topK1 = topK1_vals.logsumexp(dim=-1, keepdim=True)  # [B, T', 1]
        m_in = lse_topK1 + torch.log1p(-torch.exp(log_p - lse_topK1).clamp(min=0.0, max=1.0 - 1e-6))
        m = torch.where(is_in_topK1, m_in, lse_topK.expand_as(log_p))
        return log_p - m
