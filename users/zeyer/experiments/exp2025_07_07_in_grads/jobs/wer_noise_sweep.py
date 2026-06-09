"""WER/PER vs a model-degradation sweep, to overlay on the alignment-robustness plot:
does recognition collapse at the same level as the alignment (WBE)?

For each level: reload the model fresh, apply the chosen perturbation, greedy-decode
TIMIT val (a subset for the autoregressive Whisper), score error rate vs the reference.
- model_kind="whisper": HF WhisperForConditionalGeneration -> greedy generate -> WER (words).
- model_kind="wav2vec2ctc": HF Wav2Vec2ForCTC -> argmax CTC decode -> PER vs folded-IPA phones.

perturb_kind selects the degradation axis:
- "param":       Gaussian weight noise (discrete corruption -> expected cliff).
- "input":       Gaussian waveform noise (SNR knob -> expected smooth ramp).
- "act_noise":   Gaussian noise on every Linear output (smooth ramp).
- "act_dropout": dropout on every Linear output (smooth ramp).
"""

from __future__ import annotations
from typing import List, Optional
from sisyphus import Job, Task, tk


def _edit_distance(ref: List, hyp: List) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m]


class WerNoiseSweepJob(Job):
    """Greedy-decode error rate vs a model-degradation sweep (one model, all levels)."""

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        model_kind: str,
        model_dir: tk.Path,
        sigmas: List[float],
        seed: int = 42,
        max_seqs: Optional[int] = None,
        language: str = "en",
        returnn_root: Optional[tk.Path] = None,
        perturb_kind: str = "param",
    ):
        super().__init__()
        assert model_kind in ("whisper", "wav2vec2ctc")
        assert perturb_kind in ("param", "input", "act_noise", "act_dropout")
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.model_kind = model_kind
        self.model_dir = model_dir
        self.sigmas = sigmas
        self.seed = seed
        self.max_seqs = max_seqs
        self.language = language
        self.returnn_root = returnn_root
        self.perturb_kind = perturb_kind
        self.rqmt = {"time": 6, "cpu": 2, "gpu": 1, "mem": 24}
        self.out_wer = self.output_var("wer.txt")

    @classmethod
    def hash(cls, parsed_args):
        # backward-compat: the original param-noise jobs predate perturb_kind
        if parsed_args.get("perturb_kind") == "param":
            parsed_args = {k: v for k, v in parsed_args.items() if k != "perturb_kind"}
        return super().hash(parsed_args)

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
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.param_noise import apply_param_noise
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.perturb import (
            apply_input_noise,
            install_activation_perturbation,
        )

        import numpy as np
        import torch
        from datasets import load_dataset

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = get_content_dir_from_hub_cache_dir(self.model_dir)
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        n = len(ds) if self.max_seqs is None else min(self.max_seqs, len(ds))
        print(f"decoding {n} seqs, kind={self.model_kind}, perturb={self.perturb_kind}", flush=True)

        if self.model_kind == "whisper":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            proc = WhisperProcessor.from_pretrained(d)

            def fresh():
                return WhisperForConditionalGeneration.from_pretrained(d).to(dev).eval()

        else:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models.wav2vec2_phoneme_ctc import (
                _TIMIT61_TO_IPA,
            )

            proc = Wav2Vec2Processor.from_pretrained(d)

            def fresh():
                return Wav2Vec2ForCTC.from_pretrained(d).to(dev).eval()

        results = {}
        for sigma in [0.0] + list(self.sigmas):
            model = fresh()
            handles = []
            if self.perturb_kind == "param":
                apply_param_noise(model, sigma, self.seed)
            elif self.perturb_kind in ("act_noise", "act_dropout"):
                handles = install_activation_perturbation(model, self.perturb_kind, sigma, self.seed)
            tot_err = tot_ref = 0
            with torch.inference_mode():
                for i in range(n):
                    data = ds[i]
                    audio = np.asarray(data["audio"]["array"], dtype=np.float32)
                    if self.perturb_kind == "input":
                        audio = apply_input_noise(audio, sigma, self.seed)
                    sr = int(data["audio"]["sampling_rate"])
                    if self.model_kind == "whisper":
                        feats = proc(audio, sampling_rate=sr, return_tensors="pt").input_features.to(dev)
                        ids = model.generate(
                            feats, num_beams=1, do_sample=False, language=self.language, task="transcribe"
                        )
                        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].lower()
                        hyp = "".join(c for c in hyp if c.isalnum() or c.isspace()).split()
                        ref = [w.lower() for w in data["word_detail"]["utterance"]]
                    else:
                        iv = proc(audio, sampling_rate=sr, return_tensors="pt").input_values.to(dev)
                        pred = model(iv).logits.argmax(-1)
                        hyp = proc.batch_decode(pred)[0].split()
                        ref = [_TIMIT61_TO_IPA[p.lower()] for p in data["phonetic_detail"]["utterance"]]
                        ref = [r for r in ref if r != " "]
                    tot_err += _edit_distance(ref, hyp)
                    tot_ref += max(len(ref), 1)
            er = tot_err / tot_ref
            results[sigma] = er
            print(f"  level={sigma}: {'WER' if self.model_kind == 'whisper' else 'PER'}={er:.4f}", flush=True)
            for h in handles:
                h.remove()
            del model
            torch.cuda.empty_cache()
        self.out_wer.set(results)
