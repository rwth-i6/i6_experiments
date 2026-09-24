"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_duration_prior_jobs.py

The GIVEN phone-duration prior of ``SAE_4A_lexlat.md`` D17 / ``SAE_4A_lexlat_v2.md`` A9.

THE PRIOR (general knowledge only, no duration statistic of any alignment).  Every non-SIL type k
gets the SAME maximum-entropy law on its legal support [d_min, D_k] with mean m,
``p(d | k) ∝ exp(lambda d)`` (``model.blankfree_model.max_entropy_duration_logits``).  The SIL row
stays at the bed's uniform init.  The mean is

    m = (frame_rate / rho) x (sum of RETAINED unit frames / sum of ORIGINAL frames)

on the bed's train stream, with rho the bed's label-free target phone rate per ORIGINAL second (the
config's ``rate_rho_hz``).  rho is a rate per original second (the rate term divides the expected
non-SIL count by ``original_length``) while phi's duration d counts frames of the rVAD-masked unit
stream (only retained frames enter ``units.*.hdf``), so the retained stream's mean frames per token
is the product above.  All retained frames are spread over the phone tokens: frames phi assigns to
SIL segments are not subtracted.

:class:`BlankfreeDurationPriorMeanJob` computes m from the stream lengths alone (no label, no
alignment) and writes it with the law's summary; the model option ``reverse_duration_prior`` reads
``mean_frames`` from its json.

Cut from the source: ``DurationTableReadJob`` / ``read_checkpoint_law`` (the D17 per-epoch
duration-table read of a checkpoint; a descriptive analysis read no in-scope run needs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from sisyphus import Job, Task, tk

__all__ = ["BlankfreeDurationPriorMeanJob", "stream_frame_totals", "prior_mean_frames", "law_summary",
           "duration_law", "build_prior_summary", "render_prior_report"]

#: the thresholds of D17's reference read (P(d <= 3), P(d > 15))
SHORT_MAX = 3
LONG_MIN = 15


def stream_frame_totals(units_hdfs: Sequence[str], orig_length_hdfs: Sequence[str]) -> Dict[str, int]:
    """Sums of retained unit frames and of original 50 Hz frames over the SAME seq tags.

    ``units`` HDFs: ``seqLengths[:, 0]`` is the retained length; ``orig_length`` HDFs: one value per
    seq, the original frame count T.  The two tag sets must agree exactly, and every retained length
    must be at most its original.
    """
    import h5py

    def per_tag(paths, reader) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for path in paths:
            with h5py.File(path, "r") as fh:
                tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
                for tag, value in zip(tags, reader(fh)):
                    assert tag not in out, f"duplicate tag {tag} in {path}"
                    out[tag] = int(value)
        return out

    retained = per_tag(units_hdfs, lambda fh: np.asarray(fh["seqLengths"])[:, 0])

    def originals(fh):
        lengths = np.asarray(fh["seqLengths"])[:, 0]
        assert (lengths == 1).all(), "orig_length HDF must hold exactly one value per seq"
        return np.asarray(fh["inputs"]).reshape(-1)

    original = per_tag(orig_length_hdfs, originals)
    assert set(retained) == set(original), (len(set(retained) ^ set(original)), "tag sets differ")
    bad = [t for t in retained if not 0 < retained[t] <= original[t]]
    assert not bad, f"{len(bad)} seqs with retained length outside (0, original], e.g. {bad[:3]}"
    return {"n_seqs": len(retained), "retained_frames": int(sum(retained.values())),
            "original_frames": int(sum(original.values()))}


def prior_mean_frames(*, retained_frames: int, original_frames: int, rho_hz: float,
                      frame_rate_hz: float) -> float:
    """m = (frame_rate / rho) x (retained / original): mean retained frames per phone token."""
    assert rho_hz > 0 and frame_rate_hz > 0 and 0 < retained_frames <= original_frames
    return (float(frame_rate_hz) / float(rho_hz)) * (float(retained_frames) / float(original_frames))


def duration_law(dur_logits, cfg):
    """``[n_types, d_cap]`` float64 probabilities: phi's ``duration_log_probs`` of these logits."""
    import torch

    from ..model.reverse import SegmentalReverseModel

    logits = torch.as_tensor(dur_logits).double()
    mask = SegmentalReverseModel._build_duration_mask(cfg)
    assert tuple(logits.shape) == tuple(mask.shape), (tuple(logits.shape), tuple(mask.shape))
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1).exp()


def law_summary(probs, phones: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per type: E[d], P(d <= SHORT_MAX), P(d > LONG_MIN) of ``[n_types, d_cap]`` probabilities over d = 1..d_cap."""
    import torch

    probs = torch.as_tensor(probs).double()
    d = torch.arange(1, probs.shape[1] + 1, dtype=torch.float64)
    sums = probs.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-9), f"rows do not sum to 1: {sums}"
    out = {}
    for k, name in enumerate(phones):
        row = probs[k]
        out[name] = {"mean": float((row * d).sum()), "p_le_3": float(row[d <= SHORT_MAX].sum()),
                     "p_gt_15": float(row[d > LONG_MIN].sum())}
    return out


def _phones() -> List[str]:
    from ..phones import PHONES

    return list(PHONES)


class BlankfreeDurationPriorMeanJob(Job):
    """m of the given duration prior (module doc), from the bed's train-stream lengths.

    Label-free: reads stream lengths only.

    :param units_hdfs: the bed's rVAD-masked TRAIN unit HDFs (the stream phi scores).
    :param orig_length_hdfs: the matching ``orig_length.train.shard*.hdf`` (original frame counts).
    :param rho_hz: the bed's label-free target phone rate per ORIGINAL second (the value its
        written config's ``rate_rho_hz`` carries).
    :param frame_rate_hz: the unit clock (the lattice's ``frame_rate_hz``, 50).

    Writes ``prior.json`` (``mean_frames`` -- the key the model option reads -- the frame totals,
    rho, and the law's per-type E[d] / P(d <= 3) / P(d > 15) as the model will build it) and
    ``report.txt``.
    """

    def __init__(self, *, units_hdfs: Sequence[tk.Path], orig_length_hdfs: Sequence[tk.Path],
                 rho_hz: float, frame_rate_hz: float):
        super().__init__()
        self.units_hdfs = list(units_hdfs)
        self.orig_length_hdfs = list(orig_length_hdfs)
        self.rho_hz = float(rho_hz)
        self.frame_rate_hz = float(frame_rate_hz)
        self.out_json = self.output_path("prior.json")
        self.out_report = self.output_path("report.txt")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        summary = build_prior_summary(
            units_hdfs=[p.get_path() for p in self.units_hdfs],
            orig_length_hdfs=[p.get_path() for p in self.orig_length_hdfs],
            rho_hz=self.rho_hz, frame_rate_hz=self.frame_rate_hz)
        Path(self.out_json.get_path()).write_text(json.dumps(summary, indent=2) + "\n")
        Path(self.out_report.get_path()).write_text(render_prior_report(summary))
        print(render_prior_report(summary), flush=True)


def build_prior_summary(*, units_hdfs, orig_length_hdfs, rho_hz: float, frame_rate_hz: float) -> dict:
    """The job's whole computation, callable in-process."""
    from ..model.blankfree_model import max_entropy_duration_logits
    from ..model.reverse import ReverseConfig

    totals = stream_frame_totals(units_hdfs, orig_length_hdfs)
    mean = prior_mean_frames(retained_frames=totals["retained_frames"],
                             original_frames=totals["original_frames"],
                             rho_hz=rho_hz, frame_rate_hz=frame_rate_hz)
    assert math.isfinite(mean)
    cfg = ReverseConfig()
    phones = _phones()
    assert len(phones) == cfg.n_types and phones[cfg.sil_id] == "SIL", (len(phones), cfg.sil_id)
    logits = max_entropy_duration_logits(cfg, mean)
    logits[cfg.sil_id] = 0.0  # the SIL row stays at the bed's uniform init (zeros)
    law = law_summary(duration_law(logits, cfg), phones)
    phone_means = [law[p]["mean"] for p in phones if p != "SIL"]
    assert max(abs(v - mean) for v in phone_means) < 1e-6, (mean, min(phone_means), max(phone_means))
    return {
        "mean_frames": mean,
        "rho_hz": float(rho_hz),
        "frame_rate_hz": float(frame_rate_hz),
        "unconverted_mean_frames": float(frame_rate_hz) / float(rho_hz),
        "retained_over_original": totals["retained_frames"] / totals["original_frames"],
        **totals,
        "units_hdfs": list(units_hdfs),
        "orig_length_hdfs": list(orig_length_hdfs),
        "reverse_config": {"d_min": cfg.d_min, "d_max": cfg.d_max, "d_max_sil": cfg.d_max_sil,
                           "d_cap": cfg.d_cap, "sil_id": cfg.sil_id},
        "law": law,
        "convention": ("phone rows: max-entropy law on [d_min, D_k] with mean mean_frames "
                       "(blankfree_model.max_entropy_duration_logits); SIL row: uniform on "
                       "[d_min, D_SIL] (the bed's zero init); d counts retained unit frames"),
    }


def render_prior_report(s: dict) -> str:
    sil = s["law"]["SIL"]
    ph = s["law"]["AA"]
    return "\n".join([
        "Given phone-duration prior (SAE_4A_lexlat.md D17 / SAE_4A_lexlat_v2.md A9), label-free",
        f"train stream: {s['n_seqs']} seqs, retained unit frames {s['retained_frames']}, original "
        f"frames {s['original_frames']}, ratio {s['retained_over_original']:.6f}",
        f"rho {s['rho_hz']:.10f} /s per original second, frame rate {s['frame_rate_hz']:.1f} Hz",
        f"m = ({s['frame_rate_hz']:.1f} / rho) x ratio = {s['unconverted_mean_frames']:.6f} x "
        f"{s['retained_over_original']:.6f} = {s['mean_frames']:.6f} retained frames per phone",
        f"phone rows (all 39 identical): E[d] {ph['mean']:.6f}, P(d<=3) {ph['p_le_3']:.6f}, "
        f"P(d>15) {ph['p_gt_15']:.6f}",
        f"SIL row (uniform, trainable in both modes): E[d] {sil['mean']:.6f}, P(d<=3) "
        f"{sil['p_le_3']:.6f}, P(d>15) {sil['p_gt_15']:.6f}",
        "",
    ])
