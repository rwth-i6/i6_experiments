"""Wav2Vec2-CTC model adapter for the grad-based forced-alignment pipeline.

This is the SAME model as :class:`ForcedAlignBaselineJob` uses
(``torchaudio.pipelines.MMS_FA``: a wav2vec2 encoder with a char-level CTC
head, uroman romanization). The baseline aligns it with CTC forced-align
(``torchaudio.functional.forced_align``); this adapter instead aligns it
with *our* gradient-saliency method. Running both on one model disentangles
the two axes the MMS_FA baseline otherwise conflates -- "different model"
vs "different alignment method".

Per-token score
---------------
The other adapters (Voxtral / Phi4 / Canary) are autoregressive: the
per-token score is ``log p(y_i | y_<i, x)``, and ``log_probs`` returns a
``[B, n, V]`` distribution that the extract job gathers at the target index.
CTC is frame-synchronous and has no such per-token distribution, so we use
the CTC *partial score*

    Delta_i = log p(y_1..y_i | x) - log p(y_1..y_{i-1} | x)

(optionally including the trailing blank state), exactly the quantity the
in-house CTC grad-align used (``nn_rf.fsa.ctc_partial_scores``). We compute
all Delta_i once in ``forward`` via the standard CTC forward (alpha)
recursion over the extended (blank-interleaved) state sequence, then have
``log_probs`` scatter the requested slice into a ``[B, n, V]`` tensor at the
target indices so the existing gather-based extract loop is reused
unchanged. Grad of Delta_i w.r.t. the input feature highlights the frames
responsible for emitting token i -- the saliency we align on.

Grad target
-----------
Grads are taken w.r.t. the wav2vec2 feature-extractor output (the conv
feature encoder, ~50 Hz / 320 samples per frame), the learned-feature
analog of the log-mel grad target used for the speech LLMs, and the same
time resolution as the model's own emission (so the saliency matrix and the
CTC baseline's alignment live on the same frame grid). The encoder params
are frozen, so we leaf-ify the feature-extractor output via a forward hook
(detach -> requires_grad_ -> return), the same input-leaf trick the Voxtral
conv1 path uses.

Batch size 1 only (like the other adapters).
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict
import time

import numpy as np
import torch

from .base import BaseModelInterface, ForwardOutput


class Wav2Vec2Ctc(BaseModelInterface):
    """torchaudio MMS_FA (wav2vec2 + char CTC). See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        include_next_blank: bool = True,
        char_level_sep: Optional[str] = None,
        version: int = 1,
    ):
        """
        :param include_next_blank: if True, the partial score numerator uses
            the state *after* token i's trailing blank (``inclBlankState`` in
            the in-house CTC grad-align, its best-performing variant);
            if False, it stops at token i's label state.
        :param char_level_sep: optional separator char inserted between words
            before tokenization (e.g. ``"|"`` if the vocab has it). MMS_FA's
            uroman vocab has no space, so the default (None) simply
            concatenates each word's chars; word boundaries are recovered
            from ``target_start_end`` (per-word char ranges), not a token.
        :param version: bump only if a change alters an already-finished or
            currently-running config's output (jump past every value used so
            far). v1: initial.
        """
        super().__init__()
        assert version >= 1
        self.device = device
        self.include_next_blank = bool(include_next_blank)
        self._char_level_sep = char_level_sep
        self.version = version

        print("Import torchaudio / MMS_FA bundle...")
        start_time = time.time()
        import torchaudio  # noqa: F401
        from torchaudio.pipelines import MMS_FA as bundle

        self._bundle = bundle
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        token_dict: Dict[str, int] = bundle.get_dict()
        self.valid_chars = set(token_dict.keys())
        self.target_sr = bundle.sample_rate
        # MMS_FA blank is the '-' token at index 0 (torchaudio forced_align's
        # default blank). Look it up rather than hard-coding, but fall back to 0.
        self.blank_idx = int(token_dict.get("-", 0))
        print(f"  ({time.time() - start_time:.1f}s) vocab={len(self.valid_chars)} blank_idx={self.blank_idx}")
        print(self.model)

        # Locate the feature-extractor submodule (the conv feature encoder,
        # ~50 Hz). Its output is the grad leaf.
        fe = getattr(self.model, "feature_extractor", None)
        if fe is None:
            for m in self.model.modules():
                if type(m).__name__ == "FeatureExtractor":
                    fe = m
                    break
        assert fe is not None, "could not find wav2vec2 FeatureExtractor submodule"
        self._feature_extractor = fe

    # ---- Helpers --------------------------------------------------------

    def _norm(self, word: str) -> str:
        """Lowercase + keep only in-vocab chars (mirrors the MMS_FA baseline).

        Never returns empty; an all-out-of-vocab word collapses to the '*'
        star token so the per-word char range stays non-degenerate.
        """
        w = "".join(c for c in word.lower() if c in self.valid_chars)
        return w or "*"

    def _ctc_partial_scores(self, lp: torch.Tensor, target_ids: List[int]) -> torch.Tensor:
        """CTC forward partial scores, differentiable w.r.t. ``lp``.

        :param lp: ``[T, V]`` emission log-probs (B already squeezed).
        :param target_ids: length-S non-blank target token ids.
        :return: ``[S]`` partial scores ``Delta_i`` (grad-attached).

        Standard CTC forward over the 2S+1 extended states (blank-interleaved),
        accumulating ``logsumexp_t alpha[t, s]`` per state, then taking the
        telescoping difference (state after token i minus the blank before it),
        matching ``nn_rf.fsa.ctc_partial_scores``.
        """
        device = lp.device
        dtype = lp.dtype
        T = lp.shape[0]
        S = len(target_ids)
        blank = self.blank_idx
        # Large finite negative as the log-zero. True -inf forward-propagates
        # fine but makes the logsumexp/logaddexp/where backward produce
        # 0*inf = NaN; -1e9 underflows to weight 0 with a finite gradient.
        neg = -1.0e9

        # Extended label ids: [blank, y0, blank, y1, ..., y_{S-1}, blank].
        ext: List[int] = [blank]
        for c in target_ids:
            ext.append(int(c))
            ext.append(blank)
        sx = len(ext)  # 2S+1
        ext_t = torch.tensor(ext, device=device, dtype=torch.long)
        emit = lp[:, ext_t]  # [T, sx]: per-frame emission of each extended state's label

        # Skip transition s-2 -> s allowed only into a label state whose label
        # differs from the label two states back (standard CTC skip-blocking).
        skip_ok = torch.zeros(sx, dtype=torch.bool, device=device)
        for s in range(2, sx):
            if s % 2 == 1 and ext[s] != ext[s - 2]:
                skip_ok[s] = True

        alpha = torch.full((sx,), neg, device=device, dtype=dtype)
        alpha[0] = emit[0, 0]
        if sx > 1:
            alpha[1] = emit[0, 1]
        accum = alpha.clone()

        neg1 = torch.full((1,), neg, device=device, dtype=dtype)
        neg2 = torch.full((2,), neg, device=device, dtype=dtype)
        for t in range(1, T):
            prev = alpha
            a_stay = prev
            a_prev = torch.cat([neg1, prev[:-1]])
            a_skip = torch.cat([neg2, prev[:-2]])
            a_skip = torch.where(skip_ok, a_skip, torch.full_like(a_skip, neg))
            m = torch.logsumexp(torch.stack([a_stay, a_prev, a_skip], dim=0), dim=0)
            alpha = emit[t] + m
            accum = torch.logaddexp(accum, alpha)

        idx_label = torch.arange(S, device=device) * 2 + 1
        accum_label = accum[idx_label]
        accum_prev_blank = accum[idx_label - 1]
        accum_next_blank = accum[idx_label + 1]
        numerator = accum_next_blank if self.include_next_blank else accum_label
        return numerator - accum_prev_blank  # [S]

    # ---- Forward (forced alignment) -------------------------------------

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
        assert raw_inputs_sample_rate is not None
        assert len(raw_inputs) == 1, "Wav2Vec2Ctc wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("Wav2Vec2Ctc chunked context not implemented yet")

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])

        # Resample to the model's rate if needed. Frame->sample mapping below
        # uses orig_n_samples, so boundaries stay in the original timeline.
        wav = raw_inputs.to(dev).float()  # [1, T_samples]
        if raw_inputs_sample_rate != self.target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav, raw_inputs_sample_rate, self.target_sr)

        # Per-word char tokenization. tokenizer([w0, w1, ...]) -> list of
        # per-word token-id lists. Flatten to one target seq and record each
        # word's char range for target_start_end (word-level WBE regrouping).
        norm_words = [self._norm(w) for w in words]
        per_word_ids = self.tokenizer(norm_words)
        assert len(per_word_ids) == len(words), f"{len(per_word_ids)=} {len(words)=}"
        flat_ids: List[int] = []
        word_start_end: List[List[int]] = []
        sep_ids: List[int] = []
        if self._char_level_sep is not None:
            sep_ids = list(self.tokenizer([self._char_level_sep])[0])
        for wi, ids in enumerate(per_word_ids):
            if self._char_level_sep is not None and wi > 0:
                flat_ids.extend(sep_ids)
            cstart = len(flat_ids)
            flat_ids.extend(int(i) for i in ids)
            word_start_end.append([cstart, len(flat_ids)])
        s_len = len(flat_ids)
        assert s_len > 0, f"empty target for words={words!r}"

        # Forward with a feature-extractor leaf hook so grad flows
        # feat_extract_out -> encoder -> CTC head -> emission -> partial scores.
        captured: List[torch.Tensor] = []

        def _fe_hook(_module, _inp, out):
            x, lengths = out
            leaf = x.detach().requires_grad_(True)
            leaf.retain_grad()
            captured.append(leaf)
            return leaf, lengths

        handle = self._feature_extractor.register_forward_hook(_fe_hook)
        try:
            with torch.enable_grad():
                # MMS_FA's forward returns raw CTC logits (the bundle aligner
                # applies the log_softmax itself), so we normalize here.
                emission, _ = self.model(wav)  # [1, T_feat, V] logits
        finally:
            handle.remove()
        assert len(captured) == 1, f"expected 1 feature_extractor call, got {len(captured)}"
        grad_leaf = captured[0]  # [1, T_feat, C]
        t_feat = int(emission.shape[1])
        assert grad_leaf.shape[1] == t_feat, f"{grad_leaf.shape=} vs T_feat={t_feat}"

        log_probs = torch.log_softmax(emission.float(), dim=-1)  # [1, T_feat, V]
        vocab_size = int(log_probs.shape[-1])
        # All target ids must be real emission classes (0..V-1). The only way
        # this fails is the all-out-of-vocab '*' star fallback in _norm (id V),
        # which TIMIT never triggers; assert loudly rather than mis-index.
        assert max(flat_ids) < vocab_size, (
            f"target id {max(flat_ids)} >= vocab {vocab_size} "
            "(likely the '*' star fallback for an all-out-of-vocab word)"
        )
        partial = self._ctc_partial_scores(log_probs[0], flat_ids)  # [S], grad-attached
        # Pad one slot for the chunk-exit lookup (extract calls log_probs at
        # target index S); 0.0 is a placeholder (exit score is diagnostic only).
        partial_padded = torch.cat([partial, partial.new_zeros(1)])  # [S+1]

        targets = torch.tensor([flat_ids + [self.blank_idx]], dtype=torch.long, device=dev)  # [1, S+1]
        word_start_end = word_start_end + [[s_len, s_len + 1]]  # EOS/exit slot

        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([t_feat], dtype=torch.int64),
        )
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack(
            [edges[:-1].round().long(), edges[1:].round().long()], dim=-1
        ).unsqueeze(0)  # [1, T_feat, 2]

        print(
            f"[fwd] words={len(words)} chars={s_len} T_feat={t_feat} "
            f"transcription={''.join(norm_words)!r}",
            flush=True,
        )
        return ForwardOutput(
            inputs=grad_leaf,
            input_seq_lens=torch.tensor([t_feat]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(word_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(partial_padded=partial_padded, vocab_size=vocab_size),
        )

    # ---- log_probs -------------------------------------------------------

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Scatter the precomputed CTC partial scores into a ``[B, n, V]``
        tensor at the target indices, so the gather-based extract loop reads
        ``Delta_i`` at each target token. B=1.
        """
        partial_padded = forward_output.outputs["partial_padded"]  # [S+1], grad-attached
        v = forward_output.outputs["vocab_size"]
        start_i = int(start[0]) if isinstance(start, torch.Tensor) else int(start)
        end_i = int(end[0]) if isinstance(end, torch.Tensor) else int(end)
        n = end_i - start_i
        device = partial_padded.device

        vals = partial_padded[start_i:end_i]  # [n], grad-attached
        targets_slice = forward_output.targets[0, start_i:end_i].to(device)  # [n]
        out = partial_padded.new_zeros((1, n, v))
        out[0, torch.arange(n, device=device), targets_slice] = vals
        return out
