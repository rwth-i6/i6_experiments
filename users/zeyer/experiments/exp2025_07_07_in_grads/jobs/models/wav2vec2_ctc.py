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
        include_next_blank: Union[bool, str] = True,
        per_token_score: Optional[str] = None,
        stop_grad_blank: bool = False,
        grad_wrt: str = "feat_extract_out",
        raw_pool: int = 320,
        disable_self_attention: Optional[str] = None,
        char_level_sep: Optional[str] = None,
        audio_time_stretch: float = 1.0,
        time_stretch_method: str = "vocoder",
        version: int = 1,
    ):
        """
        :param include_next_blank: which extended states form the partial-score
            numerator / denominator (telescoping ``Δ_i = num − denom``).
            Mirrors ``nn_rf.fsa.ctc_partial_scores``. With label state ``2i+1``,
            blank-before ``2i``, blank-after ``2i+2``, prev-label ``2i-1``:

            - ``False``: num = label ``2i+1``, denom = blank-before ``2i``.
            - ``True``: num = blank-after ``2i+2``, denom = blank-before ``2i``
              (``inclBlankState``; the in-house headline variant).
            - ``"both"``: num = logaddexp(label, blank-after), denom = blank-before.
            - ``"both_prev"``: num = logaddexp(label, blank-after),
              denom = logaddexp(blank-before, prev-label).
        :param stop_grad_blank: if True, zero the gradient on the blank logit
            (``blankStopGrad``) so the diffuse blank-emission gradient doesn't
            pollute the per-token saliency. In the in-house CTC grad-align this
            barely helped alone, but combined with ``include_next_blank`` and a
            low-p reduction (L0.5/L0.1) it was the winning config.
        :param grad_wrt: differentiation target (saliency surface). All grad
            flows back to this tensor; everything below it runs without grad via
            a leaf hook. Options:
            - ``"feat_extract_out"`` (default): feature-extractor output
              (~50 Hz, 512-dim). Same tensor as ``conv6``.
            - ``"conv0"``..``"conv6"``: output of each of the 7 conv blocks.
              Cumulative strides 5,10,20,40,80,160,320 -> 3200,1600,800,400,200,
              100,50 Hz. Earlier blocks = finer time grid, more channels-mixed.
            - ``"feat_proj_layernorm"`` / ``"feat_proj_out"``: the LayerNorm
              (512-dim) and the Linear (1024-dim) of feature_projection -- the
              normalized features and the transformer input, both ~50 Hz.
            - ``"raw_waveform"``: the raw 16 kHz waveform, backprop through the
              whole conv stack. Pooled per ``raw_pool`` (see below).
        :param raw_pool: only for ``grad_wrt="raw_waveform"``. Samples per
            saliency frame: 320 -> ~50 Hz, 1 -> full 16 kHz sample-level
            alignment (no smoothing), intermediate -> higher resolution with an
            RMS-envelope smoothing. The reshape ``[1, n_frames, raw_pool]`` lets
            the extract's feature-dim reduction do the pooling.
        :param disable_self_attention: if set, bypass the transformer self-attention
            (so a frame's representation no longer mixes other frames -> grads
            stay local / token-specific). The model's posteriors/WER degrade, but
            the gradient localization may improve. ``"value"`` = use the
            current-frame value projection only (``out_proj(v_proj(x))``);
            ``"zero"`` = drop the attention sublayer entirely (residual + FFN only).
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
        assert include_next_blank in (False, True, "both", "both_prev"), include_next_blank
        self.include_next_blank = include_next_blank
        # per_token_score (shared ctc_partial modes) overrides include_next_blank when set; None = legacy.
        self.per_token_score = per_token_score
        self.stop_grad_blank = bool(stop_grad_blank)
        _valid_grad_wrt = {"feat_extract_out", "raw_waveform", "feat_proj_layernorm", "feat_proj_out"} | {
            f"conv{i}" for i in range(7)
        }
        assert grad_wrt in _valid_grad_wrt, f"{grad_wrt!r} not in {sorted(_valid_grad_wrt)}"
        self.grad_wrt = grad_wrt
        # raw_waveform pooling window (samples per saliency frame): 320 -> ~50 Hz,
        # 1 -> full 16 kHz sample-level alignment (no smoothing), intermediate ->
        # higher resolution with RMS-envelope smoothing. Only used for raw_waveform.
        assert raw_pool >= 1, raw_pool
        self.raw_pool = int(raw_pool)
        self._char_level_sep = char_level_sep
        # Pitch-preserving per-seq time-stretch (>1 = slower/longer audio).
        # The model then emits ~`audio_time_stretch`x more frames over the same speech,
        # giving a finer (sub-20ms) alignment grid;
        # the WordAlign job maps frames back via orig_audio_len / num_frames,
        # so boundaries stay in the original timeline automatically.
        self.audio_time_stretch = float(audio_time_stretch)
        assert self.audio_time_stretch > 0, audio_time_stretch
        # "vocoder" = librosa.effects.time_stretch (pitch-preserving phase vocoder, has artifacts);
        # "resample" = resample to sr*s samples fed as sr (pitch-shifted but a clean interpolation).
        assert time_stretch_method in ("vocoder", "resample"), time_stretch_method
        self.time_stretch_method = time_stretch_method
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
        # The 7 conv blocks (raw wav -> ~50 Hz; cumulative strides 5,10,20,40,
        # 80,160,320) and the feature_projection (LayerNorm 512 + Linear 512->1024),
        # for the grad_wrt sweep.
        self._conv_layers = fe.conv_layers
        self._feat_proj = None
        for m in self.model.modules():
            if type(m).__name__ == "FeatureProjection":
                self._feat_proj = m
                break
        assert self._feat_proj is not None, "could not find FeatureProjection submodule"

        self.disable_self_attention = disable_self_attention
        if disable_self_attention is not None:
            assert disable_self_attention in ("value", "zero"), disable_self_attention
            n_patched = 0
            for m in self.model.modules():
                if type(m).__name__ == "SelfAttention":
                    m.register_forward_hook(self._attn_disable_hook)
                    n_patched += 1
            assert n_patched > 0, "no SelfAttention modules found to disable"
            print(f"  disabled self-attention ({disable_self_attention}) on {n_patched} layers")

    # ---- Helpers --------------------------------------------------------

    def _attn_disable_hook(self, module, inp, out):
        """Replace a SelfAttention output: ``"value"`` -> current-frame value
        projection only (no cross-frame mixing); ``"zero"`` -> drop it."""
        x = inp[0]  # [B, T, D] input to the attention sublayer
        if self.disable_self_attention == "zero":
            new = torch.zeros_like(x)
        else:  # "value"
            new = module.out_proj(module.v_proj(x))
        return (new, *out[1:]) if isinstance(out, tuple) else new

    def _resolve_hook_target(self, grad_wrt: str):
        """Return ``(module, channels_first)`` for a hooked grad target.

        ``channels_first`` True for the conv blocks (output ``[B, C, T]``, needs
        a transpose to ``[B, T, C]``); False for the feature-extractor output
        and feature_projection points (already ``[B, T, F]``).
        """
        if grad_wrt == "feat_extract_out":
            return self._feature_extractor, False  # [B, T, 512] (~50 Hz)
        if grad_wrt == "feat_proj_layernorm":
            return self._feat_proj.layer_norm, False  # [B, T, 512], normalized
        if grad_wrt == "feat_proj_out":
            return self._feat_proj.projection, False  # [B, T, 1024], transformer input
        if grad_wrt.startswith("conv"):
            i = int(grad_wrt[len("conv") :])  # conv0..conv6
            return self._conv_layers[i], True  # [B, 512, T_i], channels-first
        raise ValueError(f"unhooked grad_wrt {grad_wrt!r}")

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
        if self.per_token_score is not None:
            from .ctc_partial import ctc_partial_scores

            return ctc_partial_scores(lp, target_ids, self.blank_idx, mode=self.per_token_score)
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
        accum_label = accum[idx_label]  # 2i+1
        accum_prev_blank = accum[idx_label - 1]  # 2i
        accum_next_blank = accum[idx_label + 1]  # 2i+2

        inb = self.include_next_blank
        if inb is False:
            numerator, denominator = accum_label, accum_prev_blank
        elif inb is True:
            numerator, denominator = accum_next_blank, accum_prev_blank
        elif inb == "both":
            numerator = torch.logaddexp(accum_label, accum_next_blank)
            denominator = accum_prev_blank
        elif inb == "both_prev":
            numerator = torch.logaddexp(accum_label, accum_next_blank)
            # prev label state 2i-1; invalid (-> log-zero) for i=0.
            idx_prev_label = idx_label - 2
            valid = idx_prev_label >= 0
            accum_prev_label = accum[idx_prev_label.clamp(min=0)]
            accum_prev_label = torch.where(valid, accum_prev_label, torch.full_like(accum_prev_label, neg))
            denominator = torch.logaddexp(accum_prev_blank, accum_prev_label)
        else:
            raise ValueError(f"invalid include_next_blank: {inb!r}")
        return numerator - denominator  # [S]

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
        if isinstance(raw_inputs, torch.Tensor) and len(raw_inputs) > 1:
            return self._forward_batched(
                raw_inputs=raw_inputs,
                raw_inputs_sample_rate=raw_inputs_sample_rate,
                raw_input_seq_lens=raw_input_seq_lens,
                raw_targets=raw_targets,
                raw_target_seq_lens=raw_target_seq_lens,
                omitted_prev_context=omitted_prev_context,
            )
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

        # Time-stretch (>1 = slower/longer) for a finer frame grid.
        # orig_n_samples stays the ORIGINAL count,
        # so the frame->time mapping (orig_audio_len / num_frames) recovers the original timeline.
        if self.audio_time_stretch != 1.0:
            if self.time_stretch_method == "resample":
                # Resample to sr*s samples and feed them as sr: slows the audio by s
                # (pitch drops) but is a clean interpolation -- no phase-vocoder smearing.
                import torchaudio

                new_sr = int(round(self.target_sr * self.audio_time_stretch))
                wav = torchaudio.functional.resample(wav, self.target_sr, new_sr)
            else:
                # Pitch-preserving phase vocoder (introduces artifacts at large stretch).
                import librosa

                audio_np = wav[0].detach().cpu().numpy().astype(np.float32)
                stretched = librosa.effects.time_stretch(audio_np, rate=1.0 / self.audio_time_stretch)
                wav = torch.tensor(stretched, dtype=wav.dtype, device=dev).unsqueeze(0)

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

        # Build the grad leaf and run the model. MMS_FA's forward returns raw
        # CTC logits (the bundle aligner applies log_softmax itself).
        if self.grad_wrt == "raw_waveform":
            # Leaf is the raw waveform, reshaped to [1, n_frames, raw_pool] so the
            # extract's feature-dim reduction collapses each raw_pool-sample block
            # to one per-frame value (RMS envelope). raw_pool=1 keeps full 16 kHz
            # sample-level resolution. Grad flows through the conv feature-extractor.
            spf = self.raw_pool
            n_blocks = int(wav.shape[1]) // spf
            assert n_blocks > 0, f"audio too short: {wav.shape=} raw_pool={spf}"
            wav_trim = wav[:, : n_blocks * spf].contiguous()
            grad_leaf = wav_trim.reshape(1, n_blocks, spf).detach().requires_grad_(True)  # [1, n_blocks, raw_pool]
            grad_leaf.retain_grad()
            with torch.enable_grad():
                emission, _ = self.model(grad_leaf.reshape(1, n_blocks * spf))  # [1, T_emit, V] logits
            t_feat = n_blocks  # saliency/alignment frame grid (not emission length)
        else:
            # Leaf-ify an intermediate activation via a forward hook. Conv blocks
            # output [B, C, T] (channels-first -> transpose to [B, T, C] so the
            # extract's feature-dim reduction collapses the channels); the
            # feature-extractor output and feature_projection points are already
            # [B, T, F]. The grad leaf's T defines the saliency/alignment grid.
            module, channels_first = self._resolve_hook_target(self.grad_wrt)
            captured: List[torch.Tensor] = []

            def _hook(_module, _inp, out):
                is_tuple = isinstance(out, tuple)
                x = out[0] if is_tuple else out
                if channels_first:
                    leaf = x.detach().transpose(1, 2).contiguous().requires_grad_(True)  # [B, T, C]
                    leaf.retain_grad()
                    captured.append(leaf)
                    x_back = leaf.transpose(1, 2)
                else:
                    leaf = x.detach().requires_grad_(True)
                    leaf.retain_grad()
                    captured.append(leaf)
                    x_back = leaf
                return (x_back, *out[1:]) if is_tuple else x_back

            handle = module.register_forward_hook(_hook)
            try:
                with torch.enable_grad():
                    emission, _ = self.model(wav)  # [1, T_emit, V] logits
            finally:
                handle.remove()
            assert len(captured) == 1, f"expected 1 hook call, got {len(captured)}"
            grad_leaf = captured[0]  # [1, T_feat, C]
            t_feat = int(grad_leaf.shape[1])  # saliency grid (= emission T only for ~50 Hz targets)

        if self.stop_grad_blank:
            # Zero the blank-logit gradient on every backward through emission
            # (registered once; persists across the extract loop's per-token
            # retain_graph backwards). Mirrors the in-house _zero_grad_blank_hook.
            _blank = self.blank_idx

            def _zero_blank_grad(grad):
                grad = grad.clone()
                grad[:, :, _blank] = 0.0
                return grad

            emission.register_hook(_zero_blank_grad)

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
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(
            0
        )  # [1, T_feat, 2]

        print(
            f"[fwd] words={len(words)} chars={s_len} T_feat={t_feat} transcription={''.join(norm_words)!r}",
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
        if forward_output.inputs.shape[0] > 1:
            return self._log_probs_batched(forward_output=forward_output, start=start, end=end)
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

    # ---- batched (B>1) variants, for seq-batched extraction ---------------

    def _forward_batched(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        """Batched (B>1) variant of :func:`forward`,
        for ``ExtractInGradsPerTokenJob(seq_batch_size>1)``.
        Valid rows match the B=1 forward exactly
        (torchaudio masks padded positions when given lengths).
        Restricted to the 50 Hz hook targets (grad grid = emission grid),
        no resampling, no time-stretch.
        """
        dev = self.device
        assert raw_inputs_sample_rate == self.target_sr, "batched: resampling not supported"
        assert self.audio_time_stretch == 1.0, "batched: time-stretch not supported"
        assert self.grad_wrt in ("feat_proj_out", "feat_proj_layernorm"), "batched: 50 Hz hook targets only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None:
            assert int(omitted_prev_context.max()) == 0
        bsz = raw_inputs.shape[0]
        wav = raw_inputs.to(dev).float()  # [B, T_samples_max], zero-padded
        lens = raw_input_seq_lens.to(dev)

        # Per-seq char tokenization (same as B=1).
        sep_ids: List[int] = []
        if self._char_level_sep is not None:
            sep_ids = list(self.tokenizer([self._char_level_sep])[0])
        per_seq_flat_ids: List[List[int]] = []
        per_seq_word_start_end: List[List[List[int]]] = []
        for b in range(bsz):
            words = list(raw_targets[b])[: int(raw_target_seq_lens[b])]
            per_word_ids = self.tokenizer([self._norm(w) for w in words])
            flat_ids: List[int] = []
            word_start_end: List[List[int]] = []
            for wi, ids in enumerate(per_word_ids):
                if self._char_level_sep is not None and wi > 0:
                    flat_ids.extend(sep_ids)
                cstart = len(flat_ids)
                flat_ids.extend(int(i) for i in ids)
                word_start_end.append([cstart, len(flat_ids)])
            assert flat_ids, f"empty target for words={words!r}"
            word_start_end.append([len(flat_ids), len(flat_ids) + 1])  # EOS/exit slot
            per_seq_flat_ids.append(flat_ids)
            per_seq_word_start_end.append(word_start_end)

        module, channels_first = self._resolve_hook_target(self.grad_wrt)
        assert not channels_first
        captured: List[torch.Tensor] = []

        def _hook(_module, _inp, out):
            is_tuple = isinstance(out, tuple)
            x = out[0] if is_tuple else out
            leaf = x.detach().requires_grad_(True)
            leaf.retain_grad()
            captured.append(leaf)
            # Non-leaf clone: torchaudio zeroes padded positions IN-PLACE downstream,
            # which is illegal on a grad-requiring leaf. Grad through clone is identity.
            x_back = leaf.clone()
            return (x_back, *out[1:]) if is_tuple else x_back

        # Replicate the torchaudio _Wav2Vec2Model wrapper batch-correctly:
        # its input layer_norm normalizes over the WHOLE tensor
        # (per-seq stats in B=1; cross-seq + padding stats for B>1 -- wrong),
        # and its star column is hardcoded to batch size 1.
        # So: normalize per seq, call the inner model, then log_softmax + star like the wrapper.
        wrapper = self.model
        assert hasattr(wrapper, "normalize_waveform"), "expected the torchaudio _Wav2Vec2Model wrapper"
        if wrapper.normalize_waveform:
            wav_n = wav.clone()
            for b in range(bsz):
                n_b = int(lens[b])
                wav_n[b, :n_b] = torch.nn.functional.layer_norm(wav[b, :n_b], (n_b,))
            wav = wav_n
        handle = module.register_forward_hook(_hook)
        try:
            with torch.enable_grad():
                emission, out_lens = wrapper.model(wav, lens)  # [B, T_emit_max, V'] logits, [B]
        finally:
            handle.remove()
        if wrapper.apply_log_softmax:
            emission = torch.nn.functional.log_softmax(emission, dim=-1)
        if wrapper.append_star:
            emission = torch.cat([emission, emission.new_zeros((*emission.shape[:2], 1))], dim=-1)
        assert len(captured) == 1, f"expected 1 hook call, got {len(captured)}"
        grad_leaf = captured[0]  # [B, T_feat_max, F]
        assert grad_leaf.shape[1] == emission.shape[1], "hook target not on the emission grid"
        t_feat = out_lens.to(torch.int64).cpu()  # [B] valid frames per seq

        if self.stop_grad_blank:
            _blank = self.blank_idx

            def _zero_blank_grad(grad):
                grad = grad.clone()
                grad[:, :, _blank] = 0.0
                return grad

            emission.register_hook(_zero_blank_grad)

        log_probs = torch.log_softmax(emission.float(), dim=-1)  # [B, T_feat_max, V]
        vocab_size = int(log_probs.shape[-1])
        s_max = max(len(ids) for ids in per_seq_flat_ids)
        partials = []
        for b in range(bsz):
            flat_ids = per_seq_flat_ids[b]
            assert max(flat_ids) < vocab_size
            partial = self._ctc_partial_scores(log_probs[b, : int(t_feat[b])], flat_ids)  # [S_b]
            # Zero-pad up to s_max+1: covers each seq's exit slot (index S_b) and beyond.
            partials.append(torch.cat([partial, partial.new_zeros(s_max + 1 - len(flat_ids))]))
        partial_padded = torch.stack(partials)  # [B, s_max+1]

        targets = torch.full((bsz, s_max + 1), self.blank_idx, dtype=torch.long, device=dev)
        for b, flat_ids in enumerate(per_seq_flat_ids):
            targets[b, : len(flat_ids)] = torch.tensor(flat_ids, dtype=torch.long, device=dev)
        w_max = max(len(wse) for wse in per_seq_word_start_end)
        target_start_end = torch.zeros((bsz, w_max, 2), dtype=torch.int64, device=dev)
        for b, wse in enumerate(per_seq_word_start_end):
            target_start_end[b, : len(wse)] = torch.tensor(wse, dtype=torch.int64, device=dev)

        t_feat_max = int(grad_leaf.shape[1])
        input_raw_start_end = torch.zeros((bsz, t_feat_max, 2), dtype=torch.int64)
        for b in range(bsz):
            n_b = int(t_feat[b])
            edges = torch.arange(n_b + 1, dtype=torch.float64) * (int(raw_input_seq_lens[b]) / max(n_b, 1))
            input_raw_start_end[b, :n_b] = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1)

        print(
            f"[fwd batched] B={bsz} words={[int(n) for n in raw_target_seq_lens]}"
            f" chars={[len(i) for i in per_seq_flat_ids]} T_feat={t_feat.tolist()}",
            flush=True,
        )
        return ForwardOutput(
            inputs=grad_leaf,
            input_seq_lens=t_feat,
            input_slice_start_end=(torch.zeros(bsz, dtype=torch.int64), t_feat),
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([len(ids) + 1 for ids in per_seq_flat_ids]),
            target_start_end=target_start_end,
            outputs=dict(partial_padded=partial_padded, vocab_size=vocab_size),
        )

    def _log_probs_batched(
        self,
        *,
        forward_output: ForwardOutput,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """B>1 variant of :func:`log_probs`: per-seq ``start``/``end`` [B],
        rows padded with zeros beyond each seq's range."""
        partial_padded = forward_output.outputs["partial_padded"]  # [B, s_max+1], grad-attached
        v = forward_output.outputs["vocab_size"]
        bsz = partial_padded.shape[0]
        device = partial_padded.device
        start = start.to(torch.int64)
        end = end.to(torch.int64)
        n = int((end - start).max())
        out = partial_padded.new_zeros((bsz, n, v))
        for b in range(bsz):
            s_b, e_b = int(start[b]), int(end[b])
            n_b = e_b - s_b
            if n_b <= 0:
                continue
            vals = partial_padded[b, s_b:e_b]  # [n_b], grad-attached
            tgt = forward_output.targets[b, s_b:e_b].to(device)
            out[b, torch.arange(n_b, device=device), tgt] = vals
        return out
