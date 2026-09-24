"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/eval_jobs.py (PairedPerDeltaJob :1382,
PAIRED_PER_* constants), src/speech_llm/sae/d8_admission.py (cluster_bootstrap :74) and
src/speech_llm/sae/emc/feature_dump.py (EXPECTED_UTTS :75).

Per-utterance paired greedy-PER delta of one greedy decode against another, with the speaker-
clustered bootstrap of D7.2.  The speaker clusters are ``gaps._clusters_by_speaker`` (s1a_job's).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk

from .per import edit_counts

__all__ = [
    "EXPECTED_UTTS",
    "PAIRED_PER_BOOT_RESAMPLES",
    "PAIRED_PER_BOOT_SEED",
    "PAIRED_PER_CLUSTER",
    "PAIRED_PER_CONVENTION",
    "cluster_bootstrap",
    "PairedPerDeltaJob",
]

#: utterances per LibriSpeech split (feature_dump.EXPECTED_UTTS)
EXPECTED_UTTS = {"dev-clean": 2703, "dev-other": 2864, "test-clean": 2620, "test-other": 2939}


def cluster_bootstrap(values, clusters, *, n_boot, seed):
    """D7.2's bootstrap, reproduced exactly: resample SPEAKERS with replacement, take every row."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_boot), dtype=np.float64)
    n_clusters = len(clusters)
    for b in range(int(n_boot)):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        boot[b] = float(values[np.concatenate([clusters[i] for i in picked])].mean())
    return boot


# =============================================================================================
# The brief's resample count and seed for this read (orchestrator 2026-09-15).  They are NOT
# emc_train_jobs.PAIRED_BOOT_RESAMPLES / PAIRED_BOOT_SEED (10,000 @ 42): that is the WER read's own
# registration, and this is a separate one, so both are named here rather than aliased.
PAIRED_PER_BOOT_RESAMPLES = 2000
PAIRED_PER_BOOT_SEED = 0
PAIRED_PER_CLUSTER = "speaker"

PAIRED_PER_CONVENTION = (
    "PairedPerDeltaJob convention (pre-registered, printed by the job and stated in its docstring):\n"
    "  * The paired unit is the UTTERANCE.  Both systems are the greedy phone decodes of the SAME\n"
    "    utterances (GreedyPerJob's own greedy_phones.json: per-frame argmax, repeats collapsed,\n"
    "    blank dropped, SIL dropped) scored against the SAME frozen MFA gold; the tag sets of A, B\n"
    "    and the gold must be IDENTICAL or the job fails -- a paired read has no partial overlap.\n"
    "  * Per utterance: d = S + D + I from eval_jobs.edit_counts (the Levenshtein GreedyPerJob\n"
    "    scores with) and N = len(gold phones).  No rescoring, no normalization variant.\n"
    "  * delta_per = (sum d_B - sum d_A) / sum N, a FRACTION in GreedyPerJob's own PER currency\n"
    "    (not percent).  A is the BASELINE argument, B is the CANDIDATE argument.\n"
    "    NEGATIVE delta_per = B is BETTER than A (fewer phone errors on the same utterances).\n"
    "  * READING RULE: 'B refines A' iff the 95 % CI of delta_per lies entirely BELOW 0, i.e. it\n"
    "    excludes 0 in B's favour.  A CI containing 0 is NOT a refinement (and licenses no claim\n"
    "    that B is worse either); a CI entirely above 0 is a degradation of B against A.\n"
    "  * CI: SPEAKERS (the LibriSpeech tag prefix before the first '-') are resampled WITH\n"
    "    replacement, every utterance of a picked speaker is taken, 2000 resamples at seed 0, and\n"
    "    the ratio-of-sums above is recomputed per resample; the interval is its [2.5, 97.5]\n"
    "    percentile.  This is PairedWerDeltaJob's construction (d8_admission.cluster_bootstrap,\n"
    "    D7.2's) at this read's own resample count and seed.\n"
    "  * A macro read (the mean of the per-utterance rate deltas (d_B - d_A) / N) is reported\n"
    "    beside it through the same primitive; the corpus-level ratio delta_per is THE number.\n"
    "  * frac_improved / frac_worse / frac_tied count utterances by d_B < / > / == d_A.\n"
)


class PairedPerDeltaJob(Job):
    __doc__ = (
        "Per-utterance paired greedy-PER delta of one greedy decode (B) against another (A).\n\n"
        + PAIRED_PER_CONVENTION
        + "\nWhy paired: better/worse claims must score the SAME items and read per-item deltas with\n"
        "a clustered bootstrap, never two pooled arm numbers (user ruling 2026-08-23).  This is the\n"
        "PER-side sibling of ``emc_train_jobs.PairedWerDeltaJob`` and exists because the word-decode\n"
        "chain the WER read needs is unusable for this phase; it reuses that job's bootstrap and\n"
        "report convention verbatim.\n\n"
        ":param per_a: the BASELINE decode -- ``GreedyPerJob.out_hyps`` (greedy_phones.json,\n"
        "    ``{seq tag: [phone, ...]}``).  ``per.json`` carries only corpus-level S/D/I/N, so the\n"
        "    per-utterance distances are recomputed here from the hypotheses and the gold.\n"
        ":param per_b: the CANDIDATE decode, same format.  With ``select_epoch`` it is instead a\n"
        "    ``{epoch: GreedyPerJob.out_hyps}`` map and the epoch is resolved at run time.\n"
        ":param gold: ``GoldPhonesJob`` json, ``{split: {seq tag: [phone, ...]}}``.\n"
        ":param split: the dev split; the utterance count is asserted against\n"
        "    ``EXPECTED_UTTS`` (dev-clean 2703, dev-other 2864).\n"
        ":param select_epoch: an ``UnsupervisedCheckpointSelectionJob.out_best_epoch`` variable.\n"
        "    Present only for the selected-checkpoint row, whose epoch is not known at graph time;\n"
        "    the picked epoch is recorded in the outputs.\n"
    )

    def __init__(
        self,
        *,
        per_a: tk.Path,
        per_b: Any,
        gold: tk.Path,
        split: str,
        name: str = "",
        baseline_name: str = "init",
        cluster: str = PAIRED_PER_CLUSTER,
        n_boot: int = PAIRED_PER_BOOT_RESAMPLES,
        seed: int = PAIRED_PER_BOOT_SEED,
        select_epoch: Optional[tk.Variable] = None,
    ):
        super().__init__()
        assert cluster == PAIRED_PER_CLUSTER, (
            f"only speaker clustering is registered for this read, got {cluster!r}; another "
            "clustering is a new registration, not a knob"
        )
        if select_epoch is None:
            self.per_b = per_b
        else:
            assert isinstance(per_b, dict) and per_b, "select_epoch needs {epoch: hyps} for per_b"
            self.per_b = {int(e): per_b[e] for e in sorted(per_b)}
        self.per_a = per_a
        self.gold = gold
        self.split = split
        self.name = name
        self.baseline_name = baseline_name
        self.cluster = cluster
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.select_epoch = select_epoch

        self.out_paired_per = self.output_path("paired_per.json")
        self.out_summary = self.output_path("summary.txt")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def distances(hyps: Dict[str, Sequence[str]], gold: Dict[str, Sequence[str]], tags: Sequence[str]):
        """``(d, N)`` arrays over ``tags``: phone edit distance S+D+I and the reference length."""
        import numpy as np

        d = np.empty(len(tags), dtype=np.float64)
        n = np.empty(len(tags), dtype=np.float64)
        for k, tag in enumerate(tags):
            ref = list(gold[tag])
            s, dl, i = edit_counts(list(hyps[tag]), ref)
            d[k] = s + dl + i
            n[k] = len(ref)
        return d, n

    def run(self):
        import json

        import numpy as np

        from .gaps import _clusters_by_speaker
        from .json_io import dump_json

        print(PAIRED_PER_CONVENTION, flush=True)

        epoch = None
        if self.select_epoch is None:
            path_b = self.per_b.get_path()
        else:
            epoch = int(self.select_epoch.get())
            assert epoch in self.per_b, f"selected epoch {epoch} not among {sorted(self.per_b)}"
            path_b = self.per_b[epoch].get_path()
            print(f"selected epoch = {epoch}  ({path_b})", flush=True)

        hyp_a = json.load(open(self.per_a.get_path()))
        hyp_b = json.load(open(path_b))
        gold_all = json.load(open(self.gold.get_path()))
        assert self.split in gold_all, f"{self.split} not in gold ({sorted(gold_all)})"
        gold = gold_all[self.split]

        # a paired read scores the SAME items: full overlap, no intersection-and-hope
        assert set(hyp_a) == set(hyp_b), (
            f"{self.split}: the two decodes cover different utterances "
            f"(only in A {len(set(hyp_a) - set(hyp_b))}, only in B {len(set(hyp_b) - set(hyp_a))})"
        )
        assert set(hyp_a) == set(gold), (
            f"{self.split}: decoded {len(hyp_a)} utterances, gold has {len(gold)}"
        )
        tags = sorted(hyp_a)
        expected = EXPECTED_UTTS.get(self.split)
        if expected is not None:
            assert len(tags) == expected, f"{self.split}: {len(tags)} utterances, expected {expected}"

        d_a, n_a = self.distances(hyp_a, gold, tags)
        d_b, n_b = self.distances(hyp_b, gold, tags)
        assert np.array_equal(n_a, n_b)  # same gold, by construction
        ref_n = n_a
        assert ref_n.sum() > 0, "zero reference phones"

        per_a = float(d_a.sum() / ref_n.sum())
        per_b = float(d_b.sum() / ref_n.sum())
        delta = per_b - per_a

        clusters = _clusters_by_speaker(tags)
        rng = np.random.default_rng(self.seed)
        boot = np.empty(self.n_boot, dtype=np.float64)
        n_clusters = len(clusters)
        for b in range(self.n_boot):
            picked = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([clusters[i] for i in picked])
            boot[b] = (d_b[idx].sum() - d_a[idx].sum()) / ref_n[idx].sum()
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

        # the macro statistic through the shared primitive, same resampling
        per_utt_delta = (d_b - d_a) / np.maximum(ref_n, 1.0)
        macro_boot = cluster_bootstrap(per_utt_delta, clusters, n_boot=self.n_boot, seed=self.seed)
        m_lo, m_hi = (float(x) for x in np.percentile(macro_boot, [2.5, 97.5]))

        n_improved = int((d_b < d_a).sum())
        n_worse = int((d_b > d_a).sum())
        n_tied = int((d_b == d_a).sum())
        refines = bool(hi < 0.0)
        record = {
            "convention": PAIRED_PER_CONVENTION,
            "split": self.split,
            "name": self.name,
            "baseline_name": self.baseline_name,
            "selected_epoch": epoch,
            "utterances": len(tags),
            "speakers": n_clusters,
            "cluster": self.cluster,
            "ref_phones": float(ref_n.sum()),
            "per_a": per_a,
            "per_b": per_b,
            "delta_per": float(delta),
            "delta_per_ci95": [lo, hi],
            "delta_per_macro": float(per_utt_delta.mean()),
            "delta_per_macro_ci95": [m_lo, m_hi],
            "frac_improved": n_improved / len(tags),
            "frac_worse": n_worse / len(tags),
            "frac_tied": n_tied / len(tags),
            "n_improved": n_improved,
            "n_worse": n_worse,
            "n_tied": n_tied,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "refines": refines,  # the reading rule: CI entirely below 0
        }
        dump_json(record, self.out_paired_per.get_path(), indent=2)

        summary = (
            f"{self.split} {self.name or 'B'} vs {self.baseline_name}"
            + (f" (selected epoch {epoch})" if epoch is not None else "")
            + f": delta_per = {delta:+.6f} [{lo:+.6f}, {hi:+.6f}] 95 % speaker-clustered bootstrap "
            f"({self.n_boot} resamples, seed {self.seed}); PER A {per_a:.6f} -> B {per_b:.6f} on "
            f"{len(tags)} utts / {n_clusters} speakers; macro {per_utt_delta.mean():+.6f} "
            f"[{m_lo:+.6f}, {m_hi:+.6f}]; improved {n_improved} / worse {n_worse} / tied {n_tied}; "
            f"negative = B better; refines = {refines}"
        )
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write(summary + "\n")
        print(summary, flush=True)
