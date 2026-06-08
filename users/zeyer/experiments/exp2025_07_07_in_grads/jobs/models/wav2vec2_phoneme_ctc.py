"""Wav2Vec2 phoneme-CTC adapter for the grad-based forced-alignment pipeline.

Model: HF ``vitouphy/wav2vec2-xls-r-300m-timit-phoneme`` -- a wav2vec2 encoder
with a CTC head over a 39-symbol IPA phone set, fine-tuned on TIMIT. This is
the "phoneme neural model" axis (the kind of model WhisperX uses for non-English
forced alignment): a model whose NATIVE label space is phonemes, so per-phone
saliency is on-distribution.

Targets here are TIMIT's own ground-truth phone labels (``phonetic_detail``),
folded 61 -> the model's 39 IPA inventory (Lee & Hon 1989 + ARPABET->IPA), so
each TIMIT phone segment is exactly one CTC target token. The extract job passes
the phone-label sequence as ``raw_targets`` (one phone per "word"); the
WordAlign job then measures PHONE-boundary error against ``phonetic_detail``.

Per-token score: the CTC partial score
``Delta_i = log p(y_1..y_i | x) - log p(y_1..y_{i-1} | x)`` (incl. the trailing
blank state), identical to :class:`Wav2Vec2Ctc`. Grad target = the wav2vec2
feature-extractor output (~50 Hz), the same surface as the char-CTC adapter.

Batch size 1 only.
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict
import time

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


# TIMIT-61 -> vitouphy 39-IPA fold (Lee & Hon 1989 collapse + ARPABET->IPA).
# Silence / closure / non-speech segments map to the space token ' ' (a real
# vocab entry), so every TIMIT segment is one alignable target unit.
_TIMIT61_TO_IPA: Dict[str, str] = {
    # monophthongs / diphthongs
    "aa": "ɑ",
    "ao": "ɑ",
    "ae": "æ",
    "ah": "ə",
    "ax": "ə",
    "ax-h": "ə",
    "aw": "aʊ",
    "ay": "aɪ",
    "eh": "ɛ",
    "er": "ɝ",
    "axr": "ɝ",
    "ey": "eɪ",
    "ih": "ɪ",
    "ix": "ɪ",
    "iy": "i",
    "ow": "oʊ",
    "oy": "ɔɪ",
    "uh": "ʊ",
    "uw": "u",
    "ux": "u",
    # consonants
    "b": "b",
    "ch": "ʧ",
    "d": "d",
    "dh": "ð",
    "dx": "ɾ",
    "nx": "n",
    "f": "f",
    "g": "g",
    "hh": "h",
    "hv": "h",
    "jh": "ʤ",
    "k": "k",
    "l": "l",
    "el": "l",
    "m": "m",
    "em": "m",
    "n": "n",
    "en": "n",
    "ng": "ŋ",
    "eng": "ŋ",
    "p": "p",
    "r": "ɹ",
    "s": "s",
    "sh": "ʃ",
    "zh": "ʃ",
    "t": "t",
    "th": "θ",
    "v": "v",
    "w": "w",
    "y": "j",
    "z": "z",
    # silence / closure / epenthetic / glottal -> space
    "bcl": " ",
    "dcl": " ",
    "gcl": " ",
    "kcl": " ",
    "pcl": " ",
    "tcl": " ",
    "epi": " ",
    "pau": " ",
    "h#": " ",
    "q": " ",
}


class Wav2Vec2PhonemeCtc(BaseModelInterface):
    """HF wav2vec2 phoneme-CTC (vitouphy TIMIT IPA). See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        include_next_blank: Union[bool, str] = True,
        grad_wrt: str = "feat_extract_out",
        param_noise_std: float = 0.0,
        param_noise_seed: int = 0,
        version: int = 1,
    ):
        """
        :param model_dir: hub-cache dir of the HF wav2vec2 phoneme-CTC model.
        :param include_next_blank: which extended states form the CTC partial
            score (see :class:`Wav2Vec2Ctc`). ``True`` = the headline variant.
        :param grad_wrt: ``"feat_extract_out"`` (default, ~50 Hz feature-extractor
            output) or ``"raw_waveform"``.
        :param version: bump only if a change alters an already-finished config.
        """
        super().__init__()
        assert version >= 1
        self.device = device
        self.model_dir = model_dir
        assert include_next_blank in (False, True, "both", "both_prev"), include_next_blank
        self.include_next_blank = include_next_blank
        assert grad_wrt in ("feat_extract_out", "raw_waveform"), grad_wrt
        self.grad_wrt = grad_wrt
        self.version = version

        print("Import / load wav2vec2 phoneme-CTC...")
        start_time = time.time()
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        d = get_content_dir_from_hub_cache_dir(model_dir)
        self.processor = Wav2Vec2Processor.from_pretrained(d)
        self.model = Wav2Vec2ForCTC.from_pretrained(d).to(device).eval()
        from ..param_noise import apply_param_noise

        apply_param_noise(self.model, param_noise_std, param_noise_seed)
        self.feature_extractor = self.processor.feature_extractor
        self.target_sr = int(self.feature_extractor.sampling_rate)
        self.vocab: Dict[str, int] = dict(self.processor.tokenizer.get_vocab())
        self.blank_idx = int(self.model.config.pad_token_id)
        # The wav2vec2 conv feature encoder (~50 Hz). Its output is the grad leaf.
        self._feat_extract = self.model.wav2vec2.feature_extractor
        # Every folded IPA symbol must be a real vocab class.
        for _ph, _ipa in _TIMIT61_TO_IPA.items():
            assert _ipa in self.vocab, f"folded IPA {_ipa!r} (from {_ph!r}) not in vocab"
        print(
            f"  ({time.time() - start_time:.1f}s) |vocab|={len(self.vocab)} blank={self.blank_idx} sr={self.target_sr}"
        )

    # ---- CTC partial scores (verbatim from Wav2Vec2Ctc) -----------------

    def _ctc_partial_scores(self, lp: torch.Tensor, target_ids: List[int]) -> torch.Tensor:
        device = lp.device
        dtype = lp.dtype
        T = lp.shape[0]
        S = len(target_ids)
        blank = self.blank_idx
        neg = -1.0e9

        ext: List[int] = [blank]
        for c in target_ids:
            ext.append(int(c))
            ext.append(blank)
        sx = len(ext)
        ext_t = torch.tensor(ext, device=device, dtype=torch.long)
        emit = lp[:, ext_t]

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
            idx_prev_label = idx_label - 2
            valid = idx_prev_label >= 0
            accum_prev_label = accum[idx_prev_label.clamp(min=0)]
            accum_prev_label = torch.where(valid, accum_prev_label, torch.full_like(accum_prev_label, neg))
            denominator = torch.logaddexp(accum_prev_blank, accum_prev_label)
        else:
            raise ValueError(f"invalid include_next_blank: {inb!r}")
        return numerator - denominator

    # ---- Forward --------------------------------------------------------

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
        assert len(raw_inputs) == 1, "Wav2Vec2PhonemeCtc supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("chunked context not implemented")

        dev = self.device
        phones = raw_targets[0]  # TIMIT-61 phone labels (one per alignment unit)
        orig_n_samples = int(raw_input_seq_lens[0])

        wav = raw_inputs.to(dev).float()  # [1, T_samples]
        if raw_inputs_sample_rate != self.target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav, raw_inputs_sample_rate, self.target_sr)

        # Each phone label -> one folded-IPA vocab id; each phone is its own
        # "word" group (target_start_end [i, i+1]).
        flat_ids: List[int] = []
        for ph in phones:
            ipa = _TIMIT61_TO_IPA.get(ph.lower())
            assert ipa is not None, f"unknown TIMIT phone label {ph!r}"
            flat_ids.append(int(self.vocab[ipa]))
        s_len = len(flat_ids)
        assert s_len > 0, f"empty phone target {phones!r}"
        phone_start_end: List[List[int]] = [[i, i + 1] for i in range(s_len)]

        # Normalize per the processor (do_normalize) and feed input_values.
        input_values = self.processor(
            wav[0].detach().cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        ).input_values.to(dev)  # [1, T_samples]

        # Leaf-ify the feature-extractor output ([B, C, T] channels-first) via a
        # forward hook, mirroring Wav2Vec2Ctc's conv path.
        if self.grad_wrt == "raw_waveform":
            grad_leaf = input_values.detach().requires_grad_(True)
            grad_leaf.retain_grad()
            with torch.enable_grad():
                emission = self.model(grad_leaf).logits  # [1, T_emit, V]
            t_feat = int(emission.shape[1])
            # raw waveform: 1 saliency frame per emission frame is not defined here;
            # we keep the emission grid as the saliency grid for grad_wrt=raw too.
            raise NotImplementedError("raw_waveform grad target not wired for phoneme adapter")
        else:
            captured: List[torch.Tensor] = []

            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out  # [B, C, T]
                leaf = x.detach().transpose(1, 2).contiguous().requires_grad_(True)  # [B, T, C]
                leaf.retain_grad()
                captured.append(leaf)
                x_back = leaf.transpose(1, 2)
                return (x_back, *out[1:]) if isinstance(out, tuple) else x_back

            handle = self._feat_extract.register_forward_hook(_hook)
            try:
                with torch.enable_grad():
                    emission = self.model(input_values).logits  # [1, T_emit, V]
            finally:
                handle.remove()
            assert len(captured) == 1, f"expected 1 hook call, got {len(captured)}"
            grad_leaf = captured[0]  # [1, T_feat, C]
            t_feat = int(grad_leaf.shape[1])

        log_probs = torch.log_softmax(emission.float(), dim=-1)  # [1, T_feat, V]
        vocab_size = int(log_probs.shape[-1])
        assert max(flat_ids) < vocab_size, f"target id {max(flat_ids)} >= vocab {vocab_size}"
        partial = self._ctc_partial_scores(log_probs[0], flat_ids)  # [S]
        partial_padded = torch.cat([partial, partial.new_zeros(1)])  # [S+1]

        targets = torch.tensor([flat_ids + [self.blank_idx]], dtype=torch.long, device=dev)  # [1, S+1]
        phone_start_end = phone_start_end + [[s_len, s_len + 1]]  # exit slot

        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([t_feat], dtype=torch.int64),
        )
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(
            0
        )  # [1, T_feat, 2]

        print(f"[fwd] phones={s_len} T_feat={t_feat} seq={' '.join(phones)!r}", flush=True)
        return ForwardOutput(
            inputs=grad_leaf,
            input_seq_lens=torch.tensor([t_feat]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(phone_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(partial_padded=partial_padded, vocab_size=vocab_size),
        )

    # ---- log_probs ------------------------------------------------------

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        partial_padded = forward_output.outputs["partial_padded"]
        v = forward_output.outputs["vocab_size"]
        start_i = int(start[0]) if isinstance(start, torch.Tensor) else int(start)
        end_i = int(end[0]) if isinstance(end, torch.Tensor) else int(end)
        n = end_i - start_i
        device = partial_padded.device
        vals = partial_padded[start_i:end_i]
        targets_slice = forward_output.targets[0, start_i:end_i].to(device)
        out = partial_padded.new_zeros((1, n, v))
        out[0, torch.arange(n, device=device), targets_slice] = vals
        return out
