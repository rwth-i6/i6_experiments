"""Parakeet-CTC (NVIDIA NeMo FastConformer-CTC) adapter for grad-based forced alignment.

Model: ``nvidia/parakeet-ctc-1.1b`` -- a PLAIN FastConformer encoder + softmax CTC head (monotonic,
non-autoregressive). The general graphemic CTC representative: unlike torchaudio MMS_FA (an aligner)
or the TIMIT-phoneme CTC (phonetic) or OWSM-CTC (self-conditioned + prompt -> front-loaded emission,
so unalignable), a plain CTC emits monotonically in time and aligns cleanly.

Reuses the ParakeetRnnt NeMo loading + log-mel grad-leaf (preprocessor output, ~100 Hz). Per-token
score is the CTC forward PARTIAL score (differentiable w.r.t. the emission log-probs), same routine
as the Wav2Vec2-CTC adapter. BPE subword scoring grouped to words via the '▁' marker. Batch 1.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Optional, Union, List

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


class ParakeetCtc(BaseModelInterface):
    """NVIDIA NeMo FastConformer-CTC (BPE). See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        overlay_path: str,
        version: int = 1,
        per_token_score: str = "raw_partial",
        char_level: bool = False,
    ):
        """:param overlay_path: NeMo env overlay to activate on sys.path (passed from the recipe).
        :param per_token_score: CTC per-token score mode (see jobs.models.ctc_partial). Default
            ``raw_partial`` keeps the original (non-telescoping) behaviour for hash-stability;
            ``prefix_diff`` is the AED-consistent prefix-score difference that grad-aligns far better.
        """
        super().__init__()
        assert version >= 1
        self.device = device
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.version = version
        self.per_token_score = per_token_score
        self.char_level = bool(char_level)

        if overlay_path not in sys.path:
            sys.path.insert(0, overlay_path)

        print("Import NeMo / EncDecCTCModelBPE (from overlay)...")
        start_time = time.time()
        import nemo
        from nemo.collections.asr.models import EncDecCTCModelBPE

        print(f"  nemo={nemo.__version__} from {nemo.__file__}")

        content = get_content_dir_from_hub_cache_dir(model_dir)
        nemo_files = glob.glob(os.path.join(content, "**", "*.nemo"), recursive=True)
        assert len(nemo_files) == 1, f"expected exactly one .nemo under {content}, got {nemo_files}"
        print(f"Restoring CTC from {nemo_files[0]}...")
        self.model = EncDecCTCModelBPE.restore_from(nemo_files[0], map_location=device)
        self.model.to(device).eval()
        self.tokenizer = self.model.tokenizer

        # NeMo CTC blank is the last class (num_classes_with_blank - 1).
        self.vocab_size = int(self.model.decoder.num_classes_with_blank)  # incl. blank
        self.blank_idx = self.vocab_size - 1
        self.target_sr = int(self.model.cfg.sample_rate)
        print(
            f"  ({time.time() - start_time:.1f}s) vocab={self.vocab_size} blank_idx={self.blank_idx} sr={self.target_sr}"
        )

    def _ctc_partial_scores(self, lp: torch.Tensor, target_ids: List[int]) -> torch.Tensor:
        """Per-token CTC partial scores via the shared routine (mode = self.per_token_score)."""
        from .ctc_partial import ctc_partial_scores

        return ctc_partial_scores(lp, target_ids, self.blank_idx, mode=self.per_token_score)

    def _tokenize_words(self, words: List[str]):
        """Encode to subwords and group into words via the '▁' (word-start) marker.
        Returns (subword_ids, word_ranges) with word_ranges[w] = (start, end) subword indices."""
        if self.char_level:
            return self._tokenize_words_charlevel(words)
        text = " ".join(w.lower() for w in words)
        ids = list(self.tokenizer.text_to_ids(text))
        pieces = self.tokenizer.ids_to_tokens(ids)
        word_ranges = []
        cur_start = 0
        for j, p in enumerate(pieces):
            if j > 0 and p.startswith("▁"):
                word_ranges.append((cur_start, j))
                cur_start = j
        word_ranges.append((cur_start, len(pieces)))
        if len(word_ranges) == len(words):
            return ids, word_ranges
        ids, word_ranges = [], []
        for w in words:
            wp = list(self.tokenizer.text_to_ids(w.lower()))
            assert wp, f"word {w!r} encoded to empty"
            word_ranges.append((len(ids), len(ids) + len(wp)))
            ids.extend(wp)
        return ids, word_ranges

    def _tokenize_words_charlevel(self, words: List[str]):
        """Char-level retokenization, IDENTICAL in form to the AED / speech-LLM char-level path:
        each character is its own BARE target token and word boundaries are a separate ``▁``
        (SentencePiece word-boundary) token BETWEEN words, so detokenizing the sequence reconstructs
        the original transcript. The AED path emits bare chars + a standalone separator token
        (e.g. Voxtral: ``s h e <space> d o g``); the SentencePiece equivalent is ``s h e ▁ d o g``
        -- NOT the per-char ``▁s ▁h ▁e`` (which would decode to spurious inter-char spaces and is
        not a valid segmentation). The per-token grad is per character; the separator token belongs
        to no word (as in AED), so ``word_ranges`` cover only each word's characters.
        A rare char with no bare single-char piece falls back to SentencePiece byte encoding
        (still reconstructs).
        Returns ``(char_ids, word_ranges)`` with the same contract as :meth:`_tokenize_words`."""
        sp = getattr(self.tokenizer, "tokenizer", None)  # raw SentencePieceProcessor inside NeMo's tokenizer
        assert sp is not None and hasattr(sp, "piece_to_id"), "expected a SentencePiece-backed NeMo tokenizer"
        unk = sp.unk_id()
        sep_id = sp.piece_to_id("\u2581")  # the standalone word-boundary piece (AED's space-separator analog)
        assert sep_id != unk, "no standalone word-boundary piece in the SentencePiece vocab"
        ids: List[int] = []
        word_ranges = []
        for wi, w in enumerate(words):
            if wi > 0:
                ids.append(sep_id)  # inter-word separator, excluded from any word range (as in AED)
            start = len(ids)
            for ch in w.lower():
                pid = sp.piece_to_id(ch)
                if pid == unk:  # no bare single-char piece -> byte-fallback (still reconstructs)
                    ids.extend(int(i) for i in sp.encode(ch, out_type=int))
                else:
                    ids.append(int(pid))
            assert len(ids) > start, f"word {w!r} produced no char tokens"
            word_ranges.append((start, len(ids)))
        return ids, word_ranges

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
        assert len(raw_inputs) == 1 and isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("ParakeetCtc chunked context not implemented")
        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])

        wav = raw_inputs[0].to(dev).float()
        if raw_inputs_sample_rate != self.target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav[None], raw_inputs_sample_rate, self.target_sr)[0]

        sub_ids, word_ranges = self._tokenize_words(words)
        assert len(word_ranges) == len(words)
        s_len = len(sub_ids)
        assert s_len > 0, f"empty target for {words!r}"

        with torch.enable_grad():
            wav_len = torch.tensor([wav.shape[0]], device=dev, dtype=torch.long)
            proc, proc_len = self.model.preprocessor(input_signal=wav[None], length=wav_len)  # [1, F, T_mel]
            leaf = proc.detach().transpose(1, 2).contiguous().requires_grad_(True)  # [1, T_mel, F]
            leaf.retain_grad()
            t_feat = int(leaf.shape[1])
            enc, enc_len = self.model.encoder(audio_signal=leaf.transpose(1, 2), length=proc_len)  # [1, H, T_enc]
            log_probs = self.model.decoder(encoder_output=enc)  # [1, T_enc, V] (NeMo CTC applies log_softmax)
            lp = log_probs[0].float()
            partial = self._ctc_partial_scores(lp, sub_ids)  # [S]

        partial_padded = torch.cat([partial, partial.new_zeros(1)])
        targets = torch.tensor([sub_ids + [self.blank_idx]], dtype=torch.long, device=dev)
        word_start_end = [[a, b] for (a, b) in word_ranges] + [[s_len, s_len + 1]]
        input_slice = (torch.tensor([0], dtype=torch.int64), torch.tensor([t_feat], dtype=torch.int64))
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(0)

        print(
            f"[fwd] words={len(words)} subwords={s_len} T_feat={t_feat} T_enc={int(enc.shape[2])} "
            f"text={' '.join(w.lower() for w in words)!r}",
            flush=True,
        )
        return ForwardOutput(
            inputs=leaf,
            input_seq_lens=torch.tensor([t_feat]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(word_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(partial_padded=partial_padded, vocab_size=self.vocab_size),
        )

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

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> List[List[str]]:
        """Greedy CTC decoding via NeMo ``transcribe``. (``max_new_tokens`` unused.)"""
        assert len(raw_inputs) == 1 and raw_inputs_sample_rate == 16000
        wav = raw_inputs[0].detach().cpu().numpy().astype("float32")
        with torch.no_grad():
            out = self.model.transcribe([wav], batch_size=1, verbose=False)
        h = out[0]
        text = h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else h)
        return [str(text).split()]

    def forced_align_words(self, *, audio: torch.Tensor, sample_rate: int, words: List[str]):
        """torchaudio CTC forced-alignment of ``words`` on this model's own emission.

        Returns per-word (start, end) in seconds. This is the 'posteriors' baseline that brackets
        grad-align from above: the model's own preferred alignment of the SAME emissions.
        """
        import torchaudio

        dev = self.device
        wav = audio.to(dev).float()
        dur = int(wav.shape[0]) / sample_rate
        if sample_rate != self.target_sr:
            wav = torchaudio.functional.resample(wav[None], sample_rate, self.target_sr)[0]
        sub_ids, word_ranges = self._tokenize_words(words)
        assert len(word_ranges) == len(words)
        with torch.no_grad():
            wav_len = torch.tensor([wav.shape[0]], device=dev, dtype=torch.long)
            proc, proc_len = self.model.preprocessor(input_signal=wav[None], length=wav_len)  # [1, F, T_mel]
            enc, enc_len = self.model.encoder(audio_signal=proc, length=proc_len)  # [1, H, T_enc]
            log_probs = self.model.decoder(encoder_output=enc).float()  # [1, T_enc, V] (already log_softmax)
        n_frames = int(log_probs.shape[1])
        # Char-level targets can exceed the CTC emission length (this FastConformer emits at ~12.5 Hz,
        # ~80 ms/frame -- far coarser than one frame per character), which torchaudio.forced_align rejects
        # ("targets length is too long for CTC"). Frame-repeat the emission to a fine-enough grid -- the
        # same frame-upsampling the grad DP applies to its score grid -- so the posteriors aligner can
        # place char-level targets. Subword targets always fit, so this never triggers for them.
        n_repeats = sum(1 for a, b in zip(sub_ids, sub_ids[1:]) if a == b)
        need = len(sub_ids) + n_repeats
        if n_frames < need:
            up = -(-need // n_frames)  # ceil division
            log_probs = log_probs.repeat_interleave(up, dim=1)
            n_frames *= up
        spf = dur / max(n_frames, 1)
        targets = torch.tensor([sub_ids], dtype=torch.int32, device=dev)
        aligned, scores = torchaudio.functional.forced_align(log_probs, targets, blank=self.blank_idx)
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=self.blank_idx)
        assert len(spans) == len(sub_ids), f"{len(spans)} vs {len(sub_ids)}"
        pred_sub_se = [(s.start * spf, s.end * spf) for s in spans]
        return [(pred_sub_se[a][0], pred_sub_se[b - 1][1]) for a, b in word_ranges]
