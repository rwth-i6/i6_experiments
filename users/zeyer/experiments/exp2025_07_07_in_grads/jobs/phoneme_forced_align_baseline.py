"""Standard CTC forced-alignment baseline on the phoneme-CTC model.

Same model as :class:`Wav2Vec2PhonemeCtc` (vitouphy wav2vec2 + 39-IPA CTC head),
but aligned with torchaudio's Viterbi CTC forced alignment
(``torchaudio.functional.forced_align``) instead of our gradient saliency. This
is the "WhisperX on its own phoneme model" point: it brackets the grad-align
phone-WBE from above (the model's own preferred alignment of the SAME emissions).

Targets = TIMIT's ground-truth phone labels (``phonetic_detail``), folded
61 -> 39 IPA (the adapter's ``_TIMIT61_TO_IPA``). Reports BOTH phone-boundary
WBE (vs the phone segments) and word-boundary WBE (phones collapsed to words),
computed in-job via the shared align metrics.
"""

from __future__ import annotations
from typing import Optional
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class ForcedAlignPhonemeBaselineJob(Job):
    """CTC forced-alignment (torchaudio) of TIMIT phones on the vitouphy model."""

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_dir: tk.Path,
        dataset_offset_factors: int,
        param_noise_std: float = 0.0,
        param_noise_seed: int = 0,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_dir = model_dir
        self.dataset_offset_factors = dataset_offset_factors
        self.param_noise_std = param_noise_std
        self.param_noise_seed = param_noise_seed
        self.returnn_root = returnn_root
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        self.out_phone_wbe = self.output_var("phone_wbe.txt")
        self.out_word_wbe = self.output_var("word_wbe.txt")
        self.out_phone_metrics = self.output_var("phone_metrics.txt")
        self.out_word_metrics = self.output_var("word_metrics.txt")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch
        import torchaudio
        from datasets import load_dataset
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_phoneme_ctc import (
            _TIMIT61_TO_IPA,
        )
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            per_utt_boundary_errors,
            aggregate_corpus,
            collapse_phones_to_words,
        )

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = get_content_dir_from_hub_cache_dir(self.model_dir)
        processor = Wav2Vec2Processor.from_pretrained(d)
        model = Wav2Vec2ForCTC.from_pretrained(d).to(dev).eval()
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.param_noise import apply_param_noise

        apply_param_noise(model, self.param_noise_std, self.param_noise_seed)
        vocab = dict(processor.tokenizer.get_vocab())
        blank = int(model.config.pad_token_id)
        target_sr = int(processor.feature_extractor.sampling_rate)
        print(f"|vocab|={len(vocab)} blank={blank} sr={target_sr}", flush=True)

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)

        phone_errs, word_errs = [], []
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            sr = int(data["audio"]["sampling_rate"])
            ph = data["phonetic_detail"]
            wd = data["word_detail"]
            phones = list(ph["utterance"])
            target_ids = [int(vocab[_TIMIT61_TO_IPA[p.lower()]]) for p in phones]

            wav = torch.tensor(audio, device=dev)[None]
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            n_orig = int(wav.shape[1])
            input_values = processor(
                wav[0].detach().cpu().numpy(), sampling_rate=target_sr, return_tensors="pt"
            ).input_values.to(dev)
            with torch.inference_mode():
                logits = model(input_values).logits  # [1, T, V]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            n_frames = int(log_probs.shape[1])
            ratio = n_orig / n_frames

            targets = torch.tensor([target_ids], dtype=torch.int32, device=dev)
            aligned, scores = torchaudio.functional.forced_align(log_probs, targets, blank=blank)
            spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=blank)
            assert len(spans) == len(target_ids), f"{len(spans)=} {len(target_ids)=} seq {seq_idx}"

            pred_phone_se = [(s.start * ratio / target_sr, s.end * ratio / target_sr) for s in spans]
            scale = self.dataset_offset_factors / target_sr
            ref_phone_se = [(s * scale, e * scale) for s, e in zip(ph["start"], ph["stop"])]
            phone_errs.append(per_utt_boundary_errors(pred_phone_se, ref_phone_se))
            pred_word_se, ref_word_se = collapse_phones_to_words(
                pred_phone_se, ph["start"], ph["stop"], wd["start"], wd["stop"], scale
            )
            word_errs.append(per_utt_boundary_errors(pred_word_se, ref_word_se))
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(phones)} phones, {len(ref_word_se)} words, n_frames={n_frames}", flush=True)

        phone_metrics = aggregate_corpus(phone_errs)
        word_metrics = aggregate_corpus(word_errs)
        print("PHONE METRICS:", phone_metrics)
        print("WORD METRICS:", word_metrics)
        self.out_phone_wbe.set(phone_metrics["wbe"])
        self.out_word_wbe.set(word_metrics["wbe"])
        self.out_phone_metrics.set(phone_metrics)
        self.out_word_metrics.set(word_metrics)
