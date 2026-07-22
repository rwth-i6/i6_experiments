"""Whisper native forced-alignment baseline (cross-attention DTW).

The Whisper-native alignment method (what whisper-timestamped / openai's word_timestamps use):
teacher-force the REFERENCE transcript through Whisper
and DTW-align its cross-attention "alignment heads" to the audio.
This is the same-model baseline for our grad-align ON Whisper (cross-attention-DTW vs gradients),
analogous to MMS_FA for CTC.
Emits per-word (start,end) seconds.

(Note: WhisperX would instead run an external wav2vec2 CTC aligner -- i.e. our MMS_FA baseline --
so it is NOT a Whisper-native method and is omitted here.)

Uses openai-whisper via the isolated overlay (no shared-env edits).
"""

from __future__ import annotations
from typing import Optional
from sisyphus import Job, Task, tk


class WhisperCrossAttnForcedAlignJob(Job):
    """Whisper cross-attention-DTW forced alignment of the reference -> word boundaries HDF."""

    # The overlay is an env path (not part of the result identity), so passing the
    # current value leaves the finished baseline's hash unchanged.
    __sis_hash_exclude__ = {"overlay": "/home/az668407/work/whisper-ts-overlay"}
    # v2: word end is the EXCLUSIVE boundary (right edge).
    # find_alignment's end = the next word's onset, the correct convention,
    # matching the grad Aligner's +1 / _dtw_spans.
    # (v1 wrongly subtracted a frame to match the then-buggy inclusive Aligner.)
    __sis_version__ = 2

    @classmethod
    def hash(cls, parsed_args):
        # keep finished param-noise/baseline jobs' hashes: the perturb kwargs are
        # hash-invisible when at their no-op default (only non-default values hash).
        parsed_args = dict(parsed_args)
        for _k in ("input_noise_std", "act_noise_std", "act_dropout", "perturb_seed", "char_level"):
            if not parsed_args.get(_k):
                parsed_args.pop(_k, None)
        return super().hash(parsed_args)

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        overlay: str,
        whisper_model: str = "base",
        char_level: bool = False,
        param_noise_std: float = 0.0,
        param_noise_seed: int = 0,
        input_noise_std: float = 0.0,
        act_noise_std: float = 0.0,
        act_dropout: float = 0.0,
        perturb_seed: int = 0,
        returnn_root: Optional[tk.Path] = None,
    ):
        """:param overlay: openai-whisper + whisper-timestamped env overlay (passed from the recipe)."""
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.overlay = overlay
        self.whisper_model = whisper_model
        self.char_level = char_level
        self.param_noise_std = param_noise_std
        self.param_noise_seed = param_noise_seed
        self.input_noise_std = input_noise_std
        self.act_noise_std = act_noise_std
        self.act_dropout = act_dropout
        self.perturb_seed = perturb_seed
        self.returnn_root = returnn_root
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        self.out_hdf = self.output_path("word_boundaries.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

        sys.path.insert(0, self.overlay)  # openai-whisper + whisper-timestamped overlay

        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch
        import whisper
        from whisper.timing import find_alignment
        from returnn.datasets.hdf import SimpleHDFWriter
        from datasets import load_dataset

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = whisper.load_model(self.whisper_model).to(dev).eval()
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.param_noise import apply_param_noise

        apply_param_noise(model, self.param_noise_std, self.param_noise_seed)
        if self.act_noise_std or self.act_dropout:
            from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.perturb import (
                install_activation_perturbation,
            )

            _kind = "act_noise" if self.act_noise_std else "act_dropout"
            install_activation_perturbation(model, _kind, self.act_noise_std or self.act_dropout, self.perturb_seed)
        tokenizer = whisper.tokenizer.get_tokenizer(
            model.is_multilingual,
            num_languages=getattr(model, "num_languages", 99),
            language="en",
            task="transcribe",
        )
        target_sr = 16000

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)
        writer = SimpleHDFWriter(self.out_hdf.get_path(), dim=2, ndim=2)

        n_skip = 0
        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            if self.input_noise_std:
                from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.perturb import apply_input_noise

                audio = apply_input_noise(audio, self.input_noise_std, self.perturb_seed)
            sr = int(data["audio"]["sampling_rate"])
            words = list(data["word_detail"]["utterance"])
            wav = torch.tensor(audio)
            if sr != target_sr:
                import torchaudio

                wav = torchaudio.functional.resample(wav[None], sr, target_sr)[0]

            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(wav), n_mels=model.dims.n_mels).to(dev)
            num_frames = int(wav.shape[0]) // 160  # real content frames
            if self.char_level:
                # One token per char (separator before each word), mirroring the grad char adapter,
                # so the cross-attn DTW runs at CHAR granularity; find_alignment regroups to words
                # by the space separators (so grad-char vs crossattn-char is an apples-to-apples
                # method comparison at identical tokenization).
                text_tokens = []
                for w in words:
                    text_tokens += tokenizer.encode(" ")
                    for ch in w.lower():
                        text_tokens += tokenizer.encode(ch)
            else:
                text = " " + " ".join(w.lower() for w in words)
                text_tokens = tokenizer.encode(text)
            alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames)

            if len(alignment) != len(words):
                # Rare (clean TIMIT text splits 1:1 on spaces).
                # Fall back to a uniform split so the per-seq word count matches the reference.
                n_skip += 1
                dur = num_frames * 0.01
                bounds = np.linspace(0, dur, len(words) + 1)
                word_se = np.stack([bounds[:-1], bounds[1:]], axis=-1).astype(np.float32)
            else:
                # find_alignment's word end is the next word's onset = the EXCLUSIVE boundary (right edge),
                # the correct word-end convention, matching the grad Aligner's +1 / _dtw_spans.
                word_se = np.array([[wt.start, max(wt.start, wt.end)] for wt in alignment], dtype=np.float32)

            tag = str(data.get("id", f"seq-{seq_idx}"))
            writer.insert_batch(word_se[None], [len(words)], [tag])
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(words)} words", flush=True)

        writer.close()
        print(f"done; word-count fallbacks: {n_skip}", flush=True)
