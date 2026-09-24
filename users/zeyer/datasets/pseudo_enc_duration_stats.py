"""
Mean phoneme duration of the pseudo-speech encoder's duration sampling
(:class:`PseudoSpeechEncoder` in ``exp2026_05_28_tts_encoder_fzj``), for the paper tables,
in 10 ms frames, with the exact sampling rule ``max(1, int(m * s * exp(sigma * eps) + 0.5))``.

:class:`SamplePseudoEncMeanDurationJob` counts it on the phoneme sequences the training sees:
text lines turned into phonemes by RETURNN's ``PhoneSeqGenerator`` with the training's ``phone_info``
(random pronunciation, random silence between words), durations sampled per phoneme.
:class:`ComputePseudoEncMeanDurationJob` is the closed-form cross-check:
the expectation per phone, weighted by the phone counts of the MFA alignment table.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

# rows of the duration table that are not phones: silence, utterance bounds, unknown, blank
NON_PHONE_LABELS = ("[space]", "[start]", "[end]", "[UNKNOWN]", "[blank]")

# the character textogram's inventory, mirroring ``_CHAR_UNITS`` in the recipe; the space is the last id
CHAR_UNITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ'"


def _char_units_seq(line: str):
    """character unit ids of ``line`` as the recipe's ``_char_units_map_seq`` builds them:
    the space id at both ends and at every word boundary, characters outside the inventory dropped"""
    sil = len(CHAR_UNITS)
    ids = [sil]
    for word in line.upper().split():
        ids.extend(CHAR_UNITS.index(ch) for ch in word if ch in CHAR_UNITS)
        ids.append(sil)
    return ids


def _medians_for(labels: Sequence[str], medians, distribution: str):
    """the per-label medians the model samples from, as :class:`PseudoSpeechEncoder` builds them"""
    import numpy as np

    m = np.asarray(medians, dtype="float64").copy()
    if distribution == "lognormal_global":
        # duration_sil_only: every speech phone gets the [start] row (the global speech median),
        # [space] keeps its own
        glob = m[list(labels).index("[start]")]
        sil = m[list(labels).index("[space]")]
        m[:] = glob
        m[list(labels).index("[space]")] = sil
    return m


def _sample_durations(rng, m, distribution: str, scale, sigma, duration_range):
    """sampled integer durations for per-position medians ``m`` (the model's training-time rule)"""
    import numpy as np

    if distribution in ("lognormal", "lognormal_global"):
        base = m * scale * np.exp(sigma * rng.standard_normal(m.shape))
        return np.maximum(np.floor(base + 0.5), 1.0)
    if distribution == "real_mean":
        return m  # the alignment's mean per label, no sampling (a reference, not a model setting)
    lo, hi = duration_range
    if distribution == "uniform":
        return rng.integers(lo, hi + 1, size=m.shape).astype("float64")
    assert distribution == "fixed" and lo == hi
    return np.full(m.shape, float(lo))


class SamplePseudoEncMeanDurationJob(Job):
    """
    Mean sampled duration per phoneme, counted on the phoneme sequences of ``num_lines`` random text lines
    as the training generates them (``phone_info`` = the training's ``PhoneSeqGenerator`` options).

    ``out_mean``: mean frames per phone ([space] / [start] / [end] excluded);
    ``out_mean_seq_frames``: mean frames per text line, i.e. the injected sequence length;
    ``distribution="real_mean"``: every symbol gets the alignment's mean duration (the reference length);
    ``out_stats``: also the mean over all symbols incl. [space], the symbol counts, and the alignment's real mean.

    ``units="chars"``: the character textogram's sequences instead (letters + apostrophe, the space
    entry at both ends and every word boundary, as ``_char_units_map_seq`` in the recipe), so only
    ``distribution`` ``"uniform"`` / ``"fixed"`` apply and ``phone_info`` / ``duration_table`` are unused.
    """

    __sis_version__ = 2  # out_mean_seq_frames added

    def __init__(
        self,
        *,
        corpus_text: tk.Path,
        phone_info: Optional[Dict[str, Any]],
        duration_table: Optional[tk.Path],
        distribution: str,
        scale: Optional[float] = None,
        sigma: Optional[float] = None,
        duration_range: Optional[Tuple[int, int]] = None,
        num_lines: int = 20_000,
        seed: int = 1,
        units: str = "phonemes",
    ):
        super().__init__()
        assert distribution in ("lognormal", "lognormal_global", "uniform", "fixed", "real_mean")
        assert units in ("phonemes", "chars")
        assert units == "phonemes" or distribution in ("uniform", "fixed"), "chars have no duration table"
        self.units = units
        self.corpus_text = corpus_text
        self.phone_info = phone_info
        self.duration_table = duration_table
        self.distribution = distribution
        self.scale = scale
        self.sigma = sigma
        self.duration_range = tuple(duration_range) if duration_range is not None else None
        self.num_lines = num_lines
        self.seed = seed
        self.out_mean = self.output_var("mean_frames.txt")
        self.out_mean_seq_frames = self.output_var("mean_seq_frames.txt")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        # 20k lines: seconds, so local like the closed-form job
        yield Task("run", mini_task=True)

    def run(self):
        import gzip
        import json
        import random
        import numpy as np
        from returnn.datasets.lm import PhoneSeqGenerator

        def _resolve(v):
            if isinstance(v, tk.Path):
                return v.get_path()
            if isinstance(v, dict):
                return {k: _resolve(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(_resolve(x) for x in v)
            return v

        if self.units == "chars":
            labels = list(CHAR_UNITS) + ["[space]"]
            to_ids = _char_units_seq
            med_per_id = np.ones(len(labels), dtype="float64")  # unused by uniform / fixed
            real_mean = None
        else:
            gen = PhoneSeqGenerator(**_resolve(self.phone_info))
            gen.random_seed(self.seed)
            labels = gen.get_class_labels()

            def to_ids(line):
                return gen.seq_to_class_idxs(gen.generate_seq(line), dtype="int32")

            npz = np.load(self.duration_table.get_path())
            tab_labels = [str(x) for x in npz["labels"]]
            means_by_label = dict(zip(tab_labels, npz["means"].astype("float64")))
            if self.distribution == "real_mean":
                med_by_label = means_by_label
            else:
                med_by_label = dict(zip(tab_labels, _medians_for(tab_labels, npz["medians"], self.distribution)))
            counts_by_label = dict(zip(tab_labels, npz["counts"].astype("float64")))
            med_per_id = np.array([med_by_label[lab] for lab in labels], dtype="float64")
            w = np.array([counts_by_label[lab] * (lab not in NON_PHONE_LABELS) for lab in tab_labels])
            real_mean = float(sum(means_by_label[lab] * wi for lab, wi in zip(tab_labels, w)) / w.sum())
        is_phone = np.array([lab not in NON_PHONE_LABELS for lab in labels])

        path = self.corpus_text.get_path()
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            lines = [ln.strip() for ln in f]
        lines = [ln for ln in lines if ln]
        random.Random(self.seed).shuffle(lines)
        lines = lines[: self.num_lines]

        rng = np.random.default_rng(self.seed)
        sum_phone = sum_all = 0.0
        n_phone = n_all = 0
        counts = np.zeros(len(labels), dtype="int64")
        for line in lines:
            ids = np.asarray(to_ids(line), dtype="int32")
            dur = _sample_durations(
                rng, med_per_id[ids], self.distribution, self.scale, self.sigma, self.duration_range
            )
            ph = is_phone[ids]
            sum_phone += float(dur[ph].sum())
            n_phone += int(ph.sum())
            sum_all += float(dur.sum())
            n_all += len(ids)
            counts += np.bincount(ids, minlength=len(labels))

        mean_phone = sum_phone / max(n_phone, 1)
        mean_seq = sum_all / max(len(lines), 1)
        self.out_mean.set(mean_phone)
        self.out_mean_seq_frames.set(mean_seq)
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(
                {
                    "mean_frames_phones": mean_phone,
                    "mean_frames_all_symbols": sum_all / max(n_all, 1),
                    "mean_frames_per_seq": mean_seq,
                    "units": self.units,
                    "num_lines": len(lines),
                    "num_phones": n_phone,
                    "num_symbols": n_all,
                    "symbol_counts": {lab: int(c) for lab, c in zip(labels, counts) if c},
                    "real_mean_frames_phones_alignment": real_mean,
                    "distribution": self.distribution,
                    "scale": self.scale,
                    "sigma": self.sigma,
                    "duration_range": self.duration_range,
                },
                f,
                indent=2,
            )
            f.write("\n")


class ComputePseudoEncMeanDurationJob(Job):
    """
    Closed-form mean sampled duration per phone (frames), count-weighted over the phones of the duration table.

    :param duration_table: ``phone_durations.npz`` of :class:`ComputeMfaPhoneDurationStatsJob`
        (``medians`` / ``means`` / ``counts`` / ``labels``)
    :param distribution: ``"lognormal"`` (per-phone median ``m``),
        ``"lognormal_global"`` (one median for all phones: the table's ``[start]`` row,
        which carries the global speech median, as ``duration_sil_only`` does),
        ``"uniform"`` (integer range, inclusive) or ``"fixed"``
    :param scale: ``s``; :param sigma: jitter; :param duration_range: for uniform / fixed
    :param num_std: integration range of ``eps`` in standard deviations (numerical expectation)
    """

    def __init__(
        self,
        *,
        duration_table: tk.Path,
        distribution: str,
        scale: Optional[float] = None,
        sigma: Optional[float] = None,
        duration_range: Optional[Tuple[int, int]] = None,
        num_std: float = 8.0,
        num_points: int = 20_001,
    ):
        super().__init__()
        assert distribution in ("lognormal", "lognormal_global", "uniform", "fixed")
        self.duration_table = duration_table
        self.distribution = distribution
        self.scale = scale
        self.sigma = sigma
        self.duration_range = tuple(duration_range) if duration_range is not None else None
        self.num_std = num_std
        self.num_points = num_points
        self.out_mean = self.output_var("mean_frames.txt")  # count-weighted over the phones
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import numpy as np

        npz = np.load(self.duration_table.get_path())
        labels: Sequence[str] = [str(x) for x in npz["labels"]]
        medians = npz["medians"].astype("float64")
        means = npz["means"].astype("float64")
        counts = npz["counts"].astype("float64")
        phone = np.array([lab not in NON_PHONE_LABELS for lab in labels])
        w = counts * phone
        w = w / w.sum()

        m = _medians_for(labels, medians, self.distribution)
        if self.distribution in ("lognormal", "lognormal_global"):
            assert self.scale is not None and self.sigma is not None
            # E over eps ~ N(0, 1) of max(1, int(m s exp(sigma eps) + 0.5)), per phone, on a grid
            eps = np.linspace(-self.num_std, self.num_std, self.num_points)
            pdf = np.exp(-0.5 * eps**2)
            pdf /= pdf.sum()
            base = m[:, None] * self.scale * np.exp(self.sigma * eps[None, :])
            sampled = np.maximum(np.floor(base + 0.5), 1.0)
            per_phone = (sampled * pdf[None, :]).sum(axis=1)
        elif self.distribution == "uniform":
            lo, hi = self.duration_range
            per_phone = np.full_like(medians, (lo + hi) / 2.0)
        else:
            lo, hi = self.duration_range
            assert lo == hi
            per_phone = np.full_like(medians, float(lo))

        mean = float((per_phone * w).sum())
        real_mean = float((means * w).sum())
        real_median_mean = float((medians * w).sum())
        self.out_mean.set(mean)
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(
                {
                    "mean_frames": mean,
                    "real_mean_frames": real_mean,
                    "real_median_weighted_frames": real_median_mean,
                    "space_median": float(medians[labels.index("[space]")]),
                    "distribution": self.distribution,
                    "scale": self.scale,
                    "sigma": self.sigma,
                    "duration_range": self.duration_range,
                    "per_phone": {lab: float(v) for lab, v, p in zip(labels, per_phone, phone) if p},
                },
                f,
                indent=2,
            )
            f.write("\n")
