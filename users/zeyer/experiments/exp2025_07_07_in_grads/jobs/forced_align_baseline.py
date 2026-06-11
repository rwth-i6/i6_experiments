"""Neural CTC forced-alignment baseline (torchaudio MMS_FA).

Forced-aligns the *reference* transcript to the audio with the MMS forced
aligner (wav2vec2-style CTC, char-level, uroman romanization), producing
per-word (start, end) in seconds. Emits a word-boundaries HDF that
:class:`CalcAlignmentMetricsFromWordBoundariesJob` consumes
(``get_data(seq, "data")`` -> ``[n_words, 2]`` seconds).

This is the WhisperX-style neural forced-align baseline. It runs in the
existing env (torchaudio is present); the MMS_FA model is cached under
TORCH_HOME (~/.cache/torch), reachable from the offline compute nodes.
"""

from __future__ import annotations
from typing import Optional
from sisyphus import Job, Task, tk


class ForcedAlignBaselineJob(Job):
    """MMS_FA forced-alignment of the reference transcript -> per-word boundaries HDF."""

    # Defaults excluded so the existing baselines keep their hash.
    __sis_hash_exclude__ = {"audio_time_stretch": 1.0, "time_stretch_method": "vocoder"}

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        audio_time_stretch: float = 1.0,
        time_stretch_method: str = "vocoder",
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.audio_time_stretch = float(audio_time_stretch)
        assert self.audio_time_stretch > 0, audio_time_stretch
        assert time_stretch_method in ("vocoder", "resample"), time_stretch_method
        self.time_stretch_method = time_stretch_method
        self.returnn_root = returnn_root
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 16}
        self.out_hdf = self.output_path("word_boundaries.hdf")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys

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
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle
        from returnn.datasets.hdf import SimpleHDFWriter
        from datasets import load_dataset

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bundle.get_model().to(dev).eval()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        _mms_dict = bundle.get_dict()
        valid_chars = set(_mms_dict.keys())
        # The aligner treats index 0 as blank and rejects it in targets. In the MMS_FA
        # uroman dict that index is the hyphen '-', which appears in spontaneous Buckeye
        # words (um-hum, twenty-one, ...); strip those chars so the targets stay blank-free.
        blank_chars = {c for c, idx in _mms_dict.items() if idx == 0}
        target_sr = bundle.sample_rate

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print("Using key:", self.dataset_key, "num seqs:", len(ds[self.dataset_key]), flush=True)

        writer = SimpleHDFWriter(self.out_hdf.get_path(), dim=2, ndim=2)

        def _norm(word: str) -> str:
            # MMS_FA tokenizer wants lowercased uroman chars (no blank-index chars);
            # never feed an empty word.
            w = "".join(c for c in word.lower() if c in valid_chars and c not in blank_chars)
            return w or "*"

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            audio = np.asarray(data["audio"]["array"], dtype=np.float32)
            sr = int(data["audio"]["sampling_rate"])
            words = list(data["word_detail"]["utterance"])

            wav = torch.tensor(audio, device=dev)[None]  # [1, T]
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            n_orig_samples = int(wav.shape[1])  # before any stretch
            # Time-stretch (>1 = slower) for a finer emission grid.
            # ratio uses the ORIGINAL sample count,
            # so the boundaries come out directly in the original timeline (no extra rescale needed).
            if self.audio_time_stretch != 1.0:
                if self.time_stretch_method == "resample":
                    # Resample to sr*s samples fed as sr: a clean interpolation (no vocoder smearing),
                    # at the cost of a pitch shift.
                    new_sr = int(round(target_sr * self.audio_time_stretch))
                    wav = torchaudio.functional.resample(wav, target_sr, new_sr)
                else:
                    import librosa

                    audio_np = wav[0].detach().cpu().numpy().astype(np.float32)
                    stretched = librosa.effects.time_stretch(audio_np, rate=1.0 / self.audio_time_stretch)
                    wav = torch.tensor(stretched, device=dev)[None]
            with torch.inference_mode():
                emission, _ = model(wav)  # [1, n_frames, n_tokens]
            n_frames = int(emission.shape[1])
            ratio = n_orig_samples / n_frames  # original samples per emission frame
            try:
                token_spans = aligner(emission[0], tokenizer([_norm(w) for w in words]))
            except RuntimeError as exc:
                # Unalignable seq: e.g. a degenerate (hallucination-looped) hyp transcript
                # with more CTC targets than emission frames. Skip the seq -- it is then
                # missing from the HDF and counted as uncovered by the metric jobs.
                print(f"WARNING: seq {seq_idx} unalignable, skipping: {exc}", flush=True)
                continue
            assert len(token_spans) == len(words), f"{len(token_spans)=} {len(words)=}"

            word_se = np.zeros((len(words), 2), dtype=np.float32)
            for wi, spans in enumerate(token_spans):
                word_se[wi, 0] = spans[0].start * ratio / target_sr
                word_se[wi, 1] = spans[-1].end * ratio / target_sr

            tag = str(data.get("id", f"seq-{seq_idx}"))
            writer.insert_batch(word_se[None], [len(words)], [tag])
            if seq_idx % 200 == 0:
                print(f"seq {seq_idx}: {len(words)} words, n_frames={n_frames}", flush=True)

        writer.close()
