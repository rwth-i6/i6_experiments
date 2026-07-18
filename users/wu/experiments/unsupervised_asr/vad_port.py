"""SAE §1.0 — rVADfast port + validation gate (silence handling for Phase-1 distribution matching).

rVADfast (utterance-level VAD) → 25 Hz frame silence masks; speech-segment timestamps for trimming.
The validation gate compares rVAD silence against MFA gold silence on dev (eval-only): frame-level
silence F1, recall bucketed by silence-run length, and speech-time-removed fraction.

Finding (dev-clean, default vad_threshold=0.4): overall silence F1 ≈ 0.78, but this is capped by
40–80 ms inter-word micro-gaps that MFA labels and any utterance VAD smooths over; recall on the
silences that matter for trimming is high (≥520 ms: ~0.84), precision ~0.83, speech removed ~14 %
(in the sane 5–25 % band). So rVAD is adequate for trimming; the literal F1≥0.85 gate is unreachable
vs MFA phone-grid silence by construction (see SAE_0.md).

Pure helpers (numpy + rVADfast) are unit-testable; the sisyphus job decodes the cached gilkeyio
parquet (audio + gold in the same rows) so audio and gold are guaranteed aligned.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FPS = 25.0
SUBFRAMES = 4  # rVADfast 10 ms frames per 40 ms (25 Hz) frame
# MFA labels treated as NON-speech (silence). 'spn' (spoken noise) is acoustically speech -> excluded.
_SIL_LABELS = {"sil", "sp", "", "<eps>", "<unk>", "noise"}


def rvad_silence(wav: np.ndarray, sr: int = 16000, vad_threshold: float = 0.4,
                 vad=None, subframes: int = SUBFRAMES) -> np.ndarray:
    """Waveform -> encoder-rate bool mask (True = silence). Aggregates rVADfast 10 ms speech labels
    (1=speech) to `subframes`-long frames; a frame is silence iff > 50 % of its subframes are
    non-speech. subframes = 100 Hz / encoder_fps (4 -> 25 Hz, 2 -> 50 Hz)."""
    from rVADfast import rVADfast

    if vad is None:
        vad = rVADfast(vad_threshold=vad_threshold)
    labels, _ = vad(np.asarray(wav, dtype=np.float32), sr)
    n = len(labels) // subframes
    if n == 0:
        return np.zeros(0, dtype=bool)
    return labels[: n * subframes].reshape(n, subframes).mean(1) < 0.5


def rvad_silence_25hz(wav: np.ndarray, sr: int = 16000, vad_threshold: float = 0.4,
                      vad=None) -> np.ndarray:
    """25 Hz (40 ms frame) silence mask -- the BEST-RQ / §1.0 validation rate."""
    return rvad_silence(wav, sr=sr, vad_threshold=vad_threshold, vad=vad, subframes=SUBFRAMES)


def mask_to_speech_segments(silence_mask: np.ndarray, fps: float = FPS) -> List[Tuple[float, float]]:
    """25 Hz silence mask -> list of (start_s, end_s) speech segments (for VAD-trimming timestamps)."""
    speech = ~silence_mask
    segs: List[Tuple[float, float]] = []
    i, T = 0, len(speech)
    while i < T:
        if speech[i]:
            j = i
            while j < T and speech[j]:
                j += 1
            segs.append((i / fps, j / fps))
            i = j
        else:
            i += 1
    return segs


def gold_silence_25hz(phonemes: Sequence[dict], num_frames: int, fps: float = FPS) -> np.ndarray:
    """MFA phones -> 25 Hz bool silence mask (True=silence): frames not covered by a real phone or spn."""
    speech = np.zeros(int(num_frames), dtype=bool)
    if num_frames <= 0:
        return ~speech
    centers = (np.arange(num_frames) + 0.5) / fps
    for p in phonemes:
        if str(p["phoneme"]).lower() in _SIL_LABELS:
            continue  # spn falls through -> counted as speech
        lo = int(np.searchsorted(centers, float(p["start"]), "left"))
        hi = int(np.searchsorted(centers, float(p["end"]), "left"))
        speech[lo:hi] = True
    return ~speech


def _run_length_recall(rvad_sil: np.ndarray, gold_sil: np.ndarray) -> Dict[str, List[int]]:
    """Per gold-silence-run: [caught_frames, total_frames] bucketed by run length (frames)."""
    buckets = {"1-2": [0, 0], "3-5": [0, 0], "6-12": [0, 0], "13+": [0, 0]}

    def blab(n):
        return "1-2" if n <= 2 else "3-5" if n <= 5 else "6-12" if n <= 12 else "13+"

    i, T = 0, len(gold_sil)
    while i < T:
        if gold_sil[i]:
            j = i
            while j < T and gold_sil[j]:
                j += 1
            b = blab(j - i)
            buckets[b][1] += j - i
            buckets[b][0] += int(rvad_sil[i:j].sum())
            i = j
        else:
            i += 1
    return buckets


def validate_records(records: Sequence[Tuple[np.ndarray, Sequence[dict]]],
                     vad_threshold: float = 0.4) -> Dict[str, object]:
    """records: (waveform, phonemes) pairs. Returns the §1.0 rVAD validation report."""
    from rVADfast import rVADfast

    vad = rVADfast(vad_threshold=vad_threshold)
    tp = fp = fn = rvad_sil_tot = tot = 0
    buckets = {"1-2": [0, 0], "3-5": [0, 0], "6-12": [0, 0], "13+": [0, 0]}
    for wav, ph in records:
        rs = rvad_silence_25hz(wav, vad=vad)
        T = min(len(rs), len(wav) // 640)
        if T < 4:
            continue
        rs = rs[:T]
        gs = gold_silence_25hz(ph, T)
        tp += int((rs & gs).sum()); fp += int((rs & ~gs).sum()); fn += int((~rs & gs).sum())
        rvad_sil_tot += int(rs.sum()); tot += T
        for b, (h, t) in _run_length_recall(rs, gs).items():
            buckets[b][0] += h; buckets[b][1] += t
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "vad_threshold": vad_threshold, "num_frames": tot,
        "silence_precision": prec, "silence_recall": rec, "silence_f1": f1,
        "gold_silence_frac": (tp + fn) / max(tot, 1),
        "speech_removed_frac": rvad_sil_tot / max(tot, 1),
        "recall_by_runlen": {b: (h / t if t else float("nan")) for b, (h, t) in buckets.items()},
    }


# ---------------------------------------------------------------------------------------------

def _decode_audio(audio_field) -> np.ndarray:
    import soundfile as sf

    wav, _ = sf.read(io.BytesIO(audio_field["bytes"]))
    return wav.mean(1) if wav.ndim > 1 else wav


def _load_gilkeyio(split: str, limit: Optional[int] = None):
    import glob, os
    import pandas as pd

    split_map = {"validation.clean": "dev_clean", "validation.other": "dev_other",
                 "test.clean": "test_clean", "test.other": "test_other"}
    hf = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
    base = os.path.join(hf, "hub", "datasets--gilkeyio--librispeech-alignments", "snapshots")
    files = sorted(glob.glob(os.path.join(base, "*", "data", f"{split_map[split]}-*.parquet")))
    assert files, f"no gilkeyio parquet for {split}"
    recs = []
    for fp in files:
        df = pd.read_parquet(fp, columns=["audio", "phonemes"])
        for r in df.itertuples():
            recs.append((_decode_audio(r.audio), r.phonemes))
            if limit and len(recs) >= limit:
                return recs
    return recs


try:
    from sisyphus import Job, Task, tk

    class RVADValidationJob(Job):
        """§1.0 rVAD port-validation gate on a dev split (eval-only, gilkeyio audio+gold)."""

        def __init__(self, split: str, vad_threshold: float = 0.4, limit: Optional[int] = None):
            self.split = split
            self.vad_threshold = vad_threshold
            self.limit = limit
            self.out_report = self.output_path("rvad_validation.json")
            self.rqmt = {"cpu": 1, "mem": 8, "time": 4}

        def tasks(self):
            yield Task("run", rqmt=self.rqmt)

        def run(self):
            import json

            recs = _load_gilkeyio(self.split, self.limit)
            report = validate_records(recs, vad_threshold=self.vad_threshold)
            report["split"] = self.split
            report["num_utts"] = len(recs)
            with open(self.out_report.get_path(), "wt") as f:
                json.dump(report, f, indent=2)

    def register_rvad_validation(splits=("validation.clean", "validation.other")):
        out = {}
        for s in splits:
            job = RVADValidationJob(s)
            job.add_alias(f"sae/1.0/rvad_validation/{s}")
            tk.register_output(f"sae/1.0/rvad_validation_{s}.json", job.out_report)
            out[s] = job.out_report
        return out

except ImportError:  # allow standalone import (no sisyphus) for the pure helpers
    pass
