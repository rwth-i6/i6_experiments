"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_eval_jobs.py (gap_reverse_config :122,
BlankfreeDerangementGapJob :136), src/speech_llm/sae/emc/blankfree_decode_gap_jobs.py
(BlankfreeDecodeGapJob, REFERENCE_GAP_TOLERANCE, SELECT_N, _median) and src/speech_llm/sae/emc/s1a_job.py
(SELECT_SEED, EDGE_SIL, with_edge_sil, select_utterances, build_derangement, speaker_of,
_clusters_by_speaker).

The derangement gap (the registered ``decoded - deranged decoded`` log p_phi per masked frame) and D4's
decode gap (gold / deranged gold / decoded under one arm's trained reverse model).

THE PORT SPLITS EACH SOURCE JOB IN THREE (brief: every model forward outside RETURNN becomes an RFJv2
plus a CPU reader).  The source jobs selected the utterances, built the derangements, loaded the
``reverse.`` checkpoint slice with ``torch.load`` and called ``reverse.evaluate`` once per condition, all
in one ``run()``.  Here:

    ReverseGapItemsJob     (CPU)  selection, feasibility and derangements exactly as the source's
                                  run(); writes items.hdf (one sequence per scored item) + items.json
    ReturnnForwardJobV2    (CPU)  reverse_score_steps: ``reverse.evaluate`` on each condition's item
                                  list, the source's call; writes scores.json {seq tag: log_p}
    BlankfreeDerangementGapJob /
    BlankfreeDecodeGapJob  (CPU)  the source's arithmetic and output files from items.json + scores

The forward runs on CPU (``device="cpu"``): ``reverse.evaluate`` builds CPU tensors, and the source jobs
requested no GPU.  :func:`derangement_gap` and :func:`decode_gap` wire the three jobs and take the source
jobs' constructor arguments.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

from .per import _hdf_sequences
from .paired import PAIRED_PER_BOOT_RESAMPLES, PAIRED_PER_BOOT_SEED, PAIRED_PER_CLUSTER

__all__ = [
    "SELECT_SEED",
    "EDGE_SIL",
    "SELECT_N",
    "REFERENCE_GAP_TOLERANCE",
    "GAP_CONDITIONS",
    "with_edge_sil",
    "select_utterances",
    "build_derangement",
    "speaker_of",
    "gap_reverse_config",
    "build_rows",
    "ReverseGapItemsJob",
    "build_reverse_score_config",
    "reverse_score_dump",
    "BlankfreeDerangementGapJob",
    "BlankfreeDecodeGapJob",
    "derangement_gap",
    "decode_gap",
]

#: the package whose relative code-object paths are hashed (``unhashed_package_root``)
_PKG = __package__.rsplit(".", 1)[0]

_alias_prefix = "sae/4a/eval"

# =============================================================================================
# s1a helpers (s1a_job.py)
# =============================================================================================
SELECT_SEED = 0  # "a deterministic seed-0 order" (see _SELECTION_NOTE)
EDGE_SIL = True  # see _EDGE_SIL_NOTE

_SELECTION_NOTE = (
    "Selection: the eligible tags of each split are ordered by RandomState(SELECT_SEED).permutation "
    "of their sorted list and the first N_UTTS_PER_SPLIT are taken. The audit job's own prefix rule "
    "(sorted()[:500], quantize_states.py:1120) would draw 8 dev-clean and 7 dev-other speakers; the "
    "seed-0 permutation draws all 40 and 33. Both the speaker-clustered bootstrap and the "
    "speaker-matched derangement need that diversity, so the seed-0 order is used."
)
_EDGE_SIL_NOTE = (
    "GoldPhonesJob's MFA strings are sil-free, but LibriSpeech segments carry leading and trailing "
    "silence and the reverse model must cover EVERY frame. One SIL token is therefore prepended and "
    "appended to every condition's string (psi_align's default sil_mode='edges', psi_align.py:66-71; "
    "it is also the only way D_sil = 50 and 'SIL may repeat' reach S1a at all). Edge SIL describes "
    "the AUDIO (leading and trailing silence), not the content, so every condition is built as a "
    "CORE string of the gold core's length and wrapped in the same two SIL tokens afterwards: the "
    "kappa substitution never touches the edges, and the unigram-draw and constant strings carry "
    "them too. The treatment is literally identical across conditions, so it cannot bias the paired "
    "read."
)


def with_edge_sil(core: Sequence[int]) -> List[int]:
    """One SIL prepended and appended (see ``_EDGE_SIL_NOTE``); identical for every condition."""
    from ..phones import SIL_ID

    return ([SIL_ID] + list(core) + [SIL_ID]) if EDGE_SIL else list(core)


def select_utterances(tags: Sequence[str], n: int, *, seed: int = SELECT_SEED) -> List[str]:
    """The first ``n`` of ``tags`` in the deterministic seed-``seed`` order (see ``_SELECTION_NOTE``)."""
    import numpy as np

    ordered = sorted(tags)
    perm = np.random.RandomState(int(seed)).permutation(len(ordered))
    return [ordered[i] for i in perm[:n]]


def build_derangement(
    entries: Sequence[Tuple[str, str, int, int]],
    *,
    is_feasible=None,
) -> Dict[str, str]:
    """``{tag: donor tag}``: another utterance of the SAME speaker with the nearest phone count.

    ``entries`` is ``[(tag, speaker, n_tokens, n_frames)]``. Never an utterance's own string. With
    ``is_feasible(tag, donor_tag)`` given, a donor whose string cannot cover this utterance's frames
    is skipped; a tag with no admissible donor is absent from the result and is dropped from the
    paired read by the caller.
    """
    by_speaker: Dict[str, List[Tuple[str, int]]] = {}
    for tag, spk, n_tok, _ in entries:
        by_speaker.setdefault(spk, []).append((tag, n_tok))
    counts = {tag: n_tok for tag, _, n_tok, _ in entries}
    out: Dict[str, str] = {}
    for tag, spk, n_tok, _ in entries:
        cands = sorted(
            (c for c in by_speaker[spk] if c[0] != tag),
            key=lambda c: (abs(c[1] - n_tok), c[0]),
        )
        for donor, _ in cands:
            if is_feasible is None or is_feasible(tag, donor):
                out[tag] = donor
                break
    assert all(out[t] != t for t in out), "an utterance was paired with itself"
    return out


def speaker_of(seq_tag: str) -> str:
    return str(seq_tag).split("-")[0]


def _clusters_by_speaker(tags: Sequence[str]):
    """Positions grouped by speaker, keys sorted -- the D7.2 cluster construction."""
    import numpy as np

    groups: Dict[str, List[int]] = {}
    for pos, tag in enumerate(tags):
        groups.setdefault(speaker_of(tag), []).append(pos)
    return [np.array(groups[k], dtype=np.int64) for k in sorted(groups)]


# =============================================================================================
# decode-gap constants (blankfree_decode_gap_jobs.py)
# =============================================================================================
#: the tolerance of the reproduction check against the banked ``derangement_gap.json`` of the same
#: arm and epoch, in nats per masked frame.  It is a CHECK OF AN IDENTITY, not a measurement: the
#: same 500 selected utterances, the same derangement, the same reverse model and the same scoring
#: call must give the same number, and the only admissible deviation is float non-associativity.
#: The dispatch names 1e-3; the observed deviation is expected to be 0.
REFERENCE_GAP_TOLERANCE = 1.0e-3

#: how many utterances the registered read selects (``BlankfreeDerangementGapJob.run``:
#: ``s1a_job.select_utterances(..., 500, seed=s1a_job.SELECT_SEED)``).  Imported as a literal here
#: because that job states it as a literal; changing it would leave the registered column.
SELECT_N = 500

#: the scored conditions, in RETURNN index order: (name, which string, whose string).  The
#: derangement gap needs the first two; D4 needs all four.
GAP_CONDITIONS = ("decoded_own", "decoded_deranged", "gold_own", "gold_deranged")


def _median(values):
    import numpy as np

    return float(np.median(np.asarray(values, dtype=np.float64))) if len(values) else float("nan")


def gap_reverse_config(reverse_kwargs=None):
    """The ``ReverseConfig`` the derangement gap rebuilds the trained phi with.

    ``reverse_kwargs`` is the model's own ``reverse_kwargs`` argument
    (``definitions/sae_emc.py``: ``rcfg = ReverseConfig(**dict(reverse_kwargs or {}))``), so an arm
    that changes the reverse model -- the k64 arm's ``n_units = 64`` observation vocabulary -- hands
    the gap the SAME dict its training config carries and the checkpoint loads.  ``None`` is the
    bed's own configuration, which is why it stays out of the job hash.
    """
    from ..model.reverse import ReverseConfig

    return ReverseConfig(**dict(reverse_kwargs or {}))


def build_rows(*, raw, gold_strings, units, eta, cfg, with_gold: bool = True):
    """``{tag: row}`` over the registered selection: both strings, z, eta, speaker, frames.

    ``BlankfreeDecodeGapJob.build_rows`` of the source.  The selection is the registered one
    (:data:`SELECT_N` utterances in ``s1a_job``'s seed-0 order over the tags with a non-empty gold
    string), so the row set of the reproduction column is the banked job's row set by construction.
    ``with_gold=False`` (the derangement gap, which never scores gold) skips the gold string; the
    decoded-string fields are the same either way.
    """
    import numpy as np

    from ..model.reverse import feasible
    from ..phones import PHONE2ID

    chosen = select_utterances([t for t, ph in gold_strings.items() if ph], SELECT_N, seed=SELECT_SEED)
    assert all(t in raw and t in units and t in eta for t in chosen)
    rows = {}
    for tag in chosen:
        y_dec = [PHONE2ID[p] for p in raw[tag]]
        z = np.asarray(units[tag], dtype=np.int64).reshape(-1)
        frames = len(z)
        row = {
            "z": z, "frames": frames, "eta": eta[tag],
            "speaker": speaker_of(tag),
            "decoded": y_dec,
            # the registered job drops an utterance whose own string cannot cover its frames;
            # the gold string has its own feasibility and is filtered separately
            "decoded_feasible": bool(y_dec) and feasible(frames, y_dec, cfg),
        }
        if with_gold:
            y_gold = with_edge_sil([PHONE2ID[p] for p in gold_strings[tag]])
            row["gold"] = y_gold
            row["gold_feasible"] = bool(y_gold) and feasible(frames, y_gold, cfg)
        rows[tag] = row
    return rows


# =============================================================================================
# 1/3: the scored items (CPU)
# =============================================================================================
class ReverseGapItemsJob(Job):
    """The items the gap reads score: selection, feasibility and derangements, as the source's run().

    :param raw_hyps: the read's ``greedy_raw.json`` (``{tag: [phone, ...]}``, SIL kept).
    :param gold: ``GoldPhonesJob`` json; selects the utterances (non-empty gold) and, with
        ``with_gold``, supplies the edge-SIL gold strings.
    :param units: the split's unit HDFs (the observations ``z``).
    :param eta_npz: ``SpeakerEtaJob``'s ``eta.npz``.
    :param reverse_kwargs: as :func:`gap_reverse_config`; the feasibility check depends on it.
    :param with_gold: ``False`` = the derangement gap's two conditions (``decoded_own``,
        ``decoded_deranged``); ``True`` = D4's four.

    Outputs ``items.hdf`` (see :mod:`.reverse_score_steps`; seq tag ``"<condition>/<tag>"``) and
    ``items.json`` (selection counts, row metadata, pairings and the ordered tags of each condition).
    """

    def __init__(self, *, raw_hyps: tk.Path, gold: tk.Path, units: Sequence[tk.Path], eta_npz: tk.Path,
                 split: str, reverse_kwargs=None, with_gold: bool = False):
        super().__init__()
        self.raw_hyps, self.gold = raw_hyps, gold
        self.units, self.eta_npz, self.split = list(units), eta_npz, str(split)
        self.reverse_kwargs = dict(reverse_kwargs) if reverse_kwargs else None
        self.with_gold = bool(with_gold)
        self.out_items = self.output_path("items.hdf")
        self.out_meta = self.output_path("items.json")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np

        from returnn.datasets.hdf import SimpleHDFWriter

        cfg = gap_reverse_config(self.reverse_kwargs)
        from ..model.reverse import feasible

        raw = json.load(open(self.raw_hyps.get_path()))
        gold_strings = json.load(open(self.gold.get_path()))[self.split]
        units = _hdf_sequences(self.units)
        eta_file = np.load(self.eta_npz.get_path(), allow_pickle=False)
        # one read of the npz member: the source indexed eta_file["eta"] per tag, which re-inflates the
        # whole array each time and keeps every copy alive through the row view (~70 GB peak in the
        # banked job); the rows are the same values
        eta_all = eta_file["eta"]
        eta = {str(t): eta_all[j] for j, t in enumerate(eta_file["tags"])}
        rows = build_rows(raw=raw, gold_strings=gold_strings, units=units, eta=eta, cfg=cfg,
                          with_gold=self.with_gold)
        chosen = list(rows)  # build_rows keeps every selected utterance, in selection order

        # the registered derangement over the DECODED strings (BlankfreeDerangementGapJob.run)
        dec_tags_all = [t for t, r in rows.items() if r["decoded_feasible"]]
        entries_dec = [(t, rows[t]["speaker"], len(rows[t]["decoded"]), rows[t]["frames"])
                       for t in dec_tags_all]
        pairing_dec = build_derangement(
            entries_dec,
            is_feasible=lambda t, donor: feasible(rows[t]["frames"], rows[donor]["decoded"], cfg))
        tags_dec = sorted(pairing_dec)
        assert tags_dec, "no utterance has an admissible decoded donor"
        conditions = {
            "decoded_own": [(t, t, "decoded") for t in tags_dec],
            "decoded_deranged": [(t, pairing_dec[t], "decoded") for t in tags_dec],
        }
        meta: Dict[str, Any] = {
            "split": self.split, "reverse_kwargs": self.reverse_kwargs, "with_gold": self.with_gold,
            "selected": len(chosen), "decoded_feasible": len(dec_tags_all),
            "pairing_decoded": pairing_dec, "tags_decoded": tags_dec,
        }
        if self.with_gold:
            # D4: gold, deranged gold on the utterances feasible under BOTH strings
            both = [t for t in tags_dec if rows[t]["gold_feasible"]]
            entries_gold = [(t, rows[t]["speaker"], len(rows[t]["gold"]), rows[t]["frames"])
                            for t in both]
            pairing_gold = build_derangement(
                entries_gold,
                is_feasible=lambda t, donor: feasible(rows[t]["frames"], rows[donor]["gold"], cfg))
            tags = sorted(pairing_gold)
            assert tags, "no utterance has an admissible gold donor"
            conditions["gold_own"] = [(t, t, "gold") for t in tags]
            conditions["gold_deranged"] = [(t, pairing_gold[t], "gold") for t in tags]
            meta.update({"pairing_gold": pairing_gold, "tags_gold": tags})
        meta["rows"] = {
            t: {"speaker": r["speaker"], "frames": int(r["frames"]), "n_decoded": len(r["decoded"]),
                "decoded_feasible": bool(r["decoded_feasible"]),
                **({"n_gold": len(r["gold"]), "gold_feasible": bool(r["gold_feasible"])}
                   if self.with_gold else {})}
            for t, r in rows.items()}
        meta["conditions"] = {name: [t for t, _, _ in items] for name, items in conditions.items()}

        writer = SimpleHDFWriter(
            filename=self.out_items.get_path(), dim=int(cfg.n_units), ndim=1,
            extra_type={"phones": (int(cfg.n_types), 1, "int32"), "eta": (1, 1, "float32"),
                        "index": (1, 1, "int32")})
        n_items = 0
        for cond_id, name in enumerate(GAP_CONDITIONS):
            if name not in conditions:
                continue
            items = conditions[name]
            for pos, (tag, owner, which) in enumerate(items):
                z = np.asarray(rows[tag]["z"].tolist(), dtype=np.int32)
                y = np.asarray(rows[owner][which], dtype=np.int32)
                e = np.asarray(rows[tag]["eta"], dtype=np.float32).reshape(-1)
                assert (z.astype(np.int64) == rows[tag]["z"]).all()
                assert np.array_equal(e, np.asarray(rows[tag]["eta"]).reshape(-1))
                idx = np.asarray([cond_id, pos, len(items)], dtype=np.int32)
                writer.insert_batch(
                    z[None, :], [len(z)], [f"{name}/{tag}"],
                    extra={"phones": y[None, :], "eta": e[None, :], "index": idx[None, :]})
                n_items += 1
        writer.close()
        meta["n_items"] = n_items
        with open(self.out_meta.get_path(), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"{self.split}: {len(chosen)} selected, {len(dec_tags_all)} decoded-feasible, "
              f"{n_items} items in {len(conditions)} conditions", flush=True)


# =============================================================================================
# 2/3: the reverse-model forward (RETURNN, CPU)
# =============================================================================================
#: one RETURNN batch must hold every item of the dataset: the forward step regroups the batch into
#: the source's per-condition ``evaluate`` calls and asserts each condition is complete.  These two
#: numbers only have to exceed the dataset (a few thousand items, a few million frames); they do
#: not enter the scores, which ``reverse.evaluate`` batches itself (8 per bucket).
REVERSE_SCORE_BATCH_SIZE = 1_000_000_000
REVERSE_SCORE_MAX_SEQS = 1_000_000


def _reverse_score_serializer(reverse_kwargs, extern_data: Dict[str, Any]):
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import, NonhashedCode, PartialImport
    from returnn.util.pprint import pformat

    return Collection(
        serializer_objects=[
            NonhashedCode(f"extern_data = {pformat(extern_data)}\n"),
            PartialImport(
                code_object_path=f"{_PKG}.analysis.reverse_score_steps.get_reverse_model",
                unhashed_package_root=_PKG,
                hashed_arguments={"reverse_kwargs": reverse_kwargs},
                unhashed_arguments={},
                import_as="get_model",
            ),
            Import(
                code_object_path=f"{_PKG}.analysis.reverse_score_steps.reverse_score_forward_step",
                unhashed_package_root=_PKG,
                import_as="forward_step",
            ),
            Import(
                code_object_path=f"{_PKG}.analysis.reverse_score_steps.ReverseScoreCallback",
                unhashed_package_root=_PKG,
                import_as="forward_callback",
            ),
        ],
        make_local_package_copy=False,
        packages=None,
    )


def build_reverse_score_config(*, items: tk.Path, reverse_kwargs=None):
    """ReturnnConfig scoring every item of ``items.hdf`` with the arm's reverse model."""
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

    from .reverse_score_steps import ITEM_KEYS

    reverse_kwargs = dict(reverse_kwargs) if reverse_kwargs else None
    data = HDFDataset(files=[items], seq_ordering="default")
    extern_data = {
        "data": {"shape": (None,), "dtype": "int32"},
        "phones": {"shape": (None,), "dtype": "int32"},
        "eta": {"shape": (None,), "dtype": "float32"},
        "index": {"shape": (None,), "dtype": "int32"},
    }
    assert tuple(extern_data) == ITEM_KEYS
    config = {
        "forward_data": data.as_returnn_opts(),
        "batch_size": REVERSE_SCORE_BATCH_SIZE,
        "max_seqs": REVERSE_SCORE_MAX_SEQS,
    }
    post_config = {"backend": "torch", "watch_memory": True}
    return ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[_reverse_score_serializer(reverse_kwargs, extern_data)],
    )


def reverse_score_dump(*, name: str, reverse_checkpoint: tk.Path, items: tk.Path, returnn_exe: tk.Path,
                       returnn_root: tk.Path, reverse_kwargs=None, time_rqmt: float = 2.0,
                       mem_rqmt: int = 16, cpu_rqmt: int = 4):
    """``ReturnnForwardJobV2`` (CPU) writing ``scores.json`` for the items of one gap read.

    ``reverse_checkpoint`` is the ``ExtractSubmoduleCheckpointJob(prefix="reverse.")`` ``model.pt``
    (``{"model", "epoch", "step"}``), which RETURNN's checkpoint loader reads as is.  The rqmt
    defaults are the source gap job's (cpu 4, mem 16, time 2).
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_core.returnn.training import PtCheckpoint

    job = ReturnnForwardJobV2(
        model_checkpoint=PtCheckpoint(reverse_checkpoint),
        returnn_config=build_reverse_score_config(items=items, reverse_kwargs=reverse_kwargs),
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["scores.json"],
        device="cpu",
        time_rqmt=time_rqmt,
        mem_rqmt=mem_rqmt,
        cpu_rqmt=cpu_rqmt,
    )
    job.add_alias(f"{_alias_prefix}/{name}/reverse_scores")
    return job


def _load_scores(meta: Dict[str, Any], scores_path: str) -> Dict[str, Dict[str, float]]:
    """``{condition: {tag: log_p}}``; every item of every condition must have been scored."""
    with open(scores_path) as fh:
        flat = json.load(fh)
    out: Dict[str, Dict[str, float]] = {}
    for name, tags in meta["conditions"].items():
        out[name] = {t: flat[f"{name}/{t}"] for t in tags}
    assert sum(len(v) for v in out.values()) == len(flat) == int(meta["n_items"]), (
        len(flat), meta["n_items"])
    return out


# =============================================================================================
# 3/3: the readers (CPU)
# =============================================================================================
class BlankfreeDerangementGapJob(Job):
    """The registered derangement gap: ``(sum own log_p - sum deranged log_p) / masked frames``.

    The source job's ``derangement_gap.json`` / ``.txt``, computed from :class:`ReverseGapItemsJob`'s
    ``items.json`` (``with_gold=False``) and the reverse-score forward's ``scores.json``; the sums run
    over the sorted matched tags in the source's order.  Build it with :func:`derangement_gap`.
    """

    def __init__(self, *, items: tk.Path, scores: tk.Path):
        super().__init__()
        self.items, self.scores = items, scores
        self.out_summary = self.output_path("derangement_gap.json")
        self.out_report = self.output_path("derangement_gap.txt")
        self.rqmt = {"cpu": 1, "mem": 2, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np

        meta = json.load(open(self.items.get_path()))
        scores = _load_scores(meta, self.scores.get_path())
        rows, pairing = meta["rows"], meta["pairing_decoded"]
        tags = sorted(pairing)
        assert tags and tags == meta["tags_decoded"]
        own, der = scores["decoded_own"], scores["decoded_deranged"]
        frames = sum(rows[t]["frames"] for t in tags)
        gap = (sum(own[t] for t in tags) - sum(der[t] for t in tags)) / frames
        assert np.isfinite(gap)
        record = {"split": meta["split"], "gap": float(gap), "matched_utterances": len(tags),
                  "selected": meta["selected"], "feasible": meta["decoded_feasible"],
                  "masked_frames": frames,
                  "own_logp_per_frame": sum(own[t] for t in tags) / frames,
                  "deranged_logp_per_frame": sum(der[t] for t in tags) / frames,
                  "pairs": [{"tag": t, "donor": pairing[t], "speaker": rows[t]["speaker"],
                             "frames": rows[t]["frames"], "own_log_p": own[t],
                             "deranged_log_p": der[t]} for t in tags]}
        with open(self.out_summary.get_path(), "w") as fh: json.dump(record, fh, indent=2)
        with open(self.out_report.get_path(), "w") as fh: fh.write(json.dumps(record, indent=2) + "\n")


class BlankfreeDecodeGapJob(Job):
    """D4: gold / deranged-gold / decoded under one arm's trained reverse model, paired.

    The source job's ``decode_gap.json``, ``per_utterance.json`` and ``summary.txt``, computed from
    :class:`ReverseGapItemsJob`'s ``items.json`` (``with_gold=True``) and the reverse-score forward's
    ``scores.json``.  Build it with :func:`decode_gap`.

    :param items: / :param scores: the two upstream outputs.
    :param reference_gap: the banked ``derangement_gap.json`` of THIS arm and epoch.  When given,
        the recomputed ``decoded - deranged decoded`` per-frame gap is asserted against its ``gap``
        field to :data:`REFERENCE_GAP_TOLERANCE`; when ``None`` the column is still computed and
        reported, and nothing is asserted.
    :param name: free text stamped into the outputs.
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        items: tk.Path,
        scores: tk.Path,
        reference_gap: Optional[tk.Path] = None,
        name: str = "",
        n_boot: int = PAIRED_PER_BOOT_RESAMPLES,
        seed: int = PAIRED_PER_BOOT_SEED,
        cluster: str = PAIRED_PER_CLUSTER,
    ):
        super().__init__()
        assert cluster == PAIRED_PER_CLUSTER, (
            f"only speaker clustering is registered for this campaign's paired reads, got "
            f"{cluster!r}; another clustering is a new registration, not a knob")
        self.items = items
        self.scores = scores
        self.reference_gap = reference_gap
        self.name = str(name)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.cluster = cluster
        self.out_summary = self.output_path("decode_gap.json")
        self.out_per_utterance = self.output_path("per_utterance.json")
        self.out_report = self.output_path("summary.txt")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np

        from .paired import cluster_bootstrap

        meta = json.load(open(self.items.get_path()))
        assert meta["with_gold"], "the decode gap needs the items of ReverseGapItemsJob(with_gold=True)"
        split = str(meta["split"])
        scores = _load_scores(meta, self.scores.get_path())
        rows = meta["rows"]
        own_dec, der_dec = scores["decoded_own"], scores["decoded_deranged"]
        gold_own, gold_der = scores["gold_own"], scores["gold_deranged"]
        pairing_dec, pairing_gold = meta["pairing_decoded"], meta["pairing_gold"]

        # --- (1) the REPRODUCTION of the registered column: decoded vs deranged decoded ----------
        tags_dec = sorted(pairing_dec)
        assert tags_dec and tags_dec == meta["tags_decoded"]
        frames_dec = sum(rows[t]["frames"] for t in tags_dec)
        registered_gap = (sum(own_dec[t] for t in tags_dec)
                          - sum(der_dec[t] for t in tags_dec)) / frames_dec
        assert np.isfinite(registered_gap)
        reference = None
        if self.reference_gap is not None:
            with open(self.reference_gap.get_path()) as fh:
                reference = json.load(fh)
            assert str(reference["split"]) == split, (reference["split"], split)
            assert int(reference["matched_utterances"]) == len(tags_dec), (
                f"the registered read matched {reference['matched_utterances']} utterances and "
                f"this one {len(tags_dec)}; the derangement is not the registered one")
            dev = abs(float(reference["gap"]) - registered_gap)
            assert dev <= REFERENCE_GAP_TOLERANCE, (
                f"the recomputed decoded-vs-deranged-decoded gap is {registered_gap:.6f} and the "
                f"banked derangement_gap.json says {float(reference['gap']):.6f} "
                f"(|dev| = {dev:.2e} > {REFERENCE_GAP_TOLERANCE:g}); this job is not scoring what "
                "the registered read scored")

        # --- (2) the D4 read: gold, deranged gold and the decode on ONE item set -----------------
        tags = sorted(pairing_gold)
        assert tags and tags == meta["tags_gold"], "no utterance has an admissible gold donor"

        n_frames = np.array([rows[t]["frames"] for t in tags], dtype=np.float64)
        n_tokens = np.array([rows[t]["n_gold"] for t in tags], dtype=np.float64)
        lp_gold = np.array([gold_own[t] for t in tags], dtype=np.float64)
        lp_der = np.array([gold_der[t] for t in tags], dtype=np.float64)
        lp_dec = np.array([own_dec[t] for t in tags], dtype=np.float64)
        assert np.isfinite(lp_gold).all() and np.isfinite(lp_der).all() and np.isfinite(lp_dec).all()

        clusters = _clusters_by_speaker(tags)

        def column(values):
            """Mean / median / bootstrap CI of one per-utterance delta vector, three ways."""
            out = {}
            for scale, denominator in (("per_utterance", np.ones_like(values)),
                                       ("per_gold_token", n_tokens),
                                       ("per_frame", n_frames)):
                x = values / denominator
                boot = cluster_bootstrap(x, clusters, n_boot=self.n_boot, seed=self.seed)
                lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
                out[scale] = {"mean": float(x.mean()), "median": _median(x),
                              "ci95": [lo, hi], "n": int(len(x))}
            return out

        record = {
            "name": self.name,
            "split": split,
            "cluster": self.cluster,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "selected": SELECT_N,
            "decoded_feasible": int(meta["decoded_feasible"]),
            "matched_decoded": len(tags_dec),
            "utterances": len(tags),
            "speakers": len(clusters),
            "masked_frames": float(n_frames.sum()),
            "gold_tokens": float(n_tokens.sum()),
            # the registered column, recomputed here and checked against its own json
            "registered_decoded_gap_per_frame": float(registered_gap),
            "registered_reference": (None if reference is None else
                                     {"gap": float(reference["gap"]),
                                      "matched_utterances": int(reference["matched_utterances"]),
                                      "path": self.reference_gap.get_path()}),
            "reference_gap_tolerance": REFERENCE_GAP_TOLERANCE,
            "mean_log_p": {"gold": float(lp_gold.mean()), "deranged_gold": float(lp_der.mean()),
                           "decoded": float(lp_dec.mean())},
            "log_p_per_frame": {"gold": float(lp_gold.sum() / n_frames.sum()),
                                "deranged_gold": float(lp_der.sum() / n_frames.sum()),
                                "decoded": float(lp_dec.sum() / n_frames.sum())},
            "contrasts": {
                "gold_minus_decoded": column(lp_gold - lp_dec),
                "deranged_gold_minus_decoded": column(lp_der - lp_dec),
                "gold_minus_deranged_gold": column(lp_gold - lp_der),
            },
            "frac_decoded_above_gold": float((lp_dec > lp_gold).mean()),
            "n_decoded_above_gold": int((lp_dec > lp_gold).sum()),
            "conventions": {
                "gold_string": "GoldPhonesJob phones wrapped in one SIL at each end "
                               "(s1a_job.with_edge_sil / _EDGE_SIL_NOTE)",
                "decoded_string": "greedy_raw.json: run-collapsed per-frame argmax, SIL kept",
                "gold_derangement": "s1a_job.build_derangement over the GOLD strings (same "
                                    "speaker, nearest gold phone count, donor string must cover "
                                    "this utterance's frames)",
                "registered_column": "BlankfreeDerangementGapJob scores the DECODE and a deranged "
                                     "DECODE, so its banked gap is decoded - deranged decoded per "
                                     "masked frame; it is reproduced above, not re-defined",
                "per_gold_token": "the delta divided by the gold string's token count (the one "
                                  "denominator the three strings share)",
                "sign": "positive gold_minus_decoded = the reverse model prefers the GOLD string",
            },
            "per_utterance_file": self.out_per_utterance.get_path(),
        }
        per_utt = {
            t: {
                "speaker": rows[t]["speaker"],
                "frames": int(rows[t]["frames"]),
                "gold_donor": pairing_gold[t],
                "decoded_donor": pairing_dec[t],
                "tokens": {"gold": rows[t]["n_gold"], "decoded": rows[t]["n_decoded"],
                           "deranged_gold": rows[pairing_gold[t]]["n_gold"]},
                "log_p": {"gold": gold_own[t], "deranged_gold": gold_der[t],
                          "decoded": own_dec[t], "deranged_decoded": der_dec[t]},
            }
            for t in tags
        }
        with open(self.out_per_utterance.get_path(), "w") as fh:
            json.dump(per_utt, fh)
        with open(self.out_summary.get_path(), "w") as fh:
            json.dump(record, fh, indent=2)

        lines = [
            f"{self.name or 'decode gap'}  ({split}, D4 of SAE_4A_lexlat.md -- a READ, no gate)",
            f"utterances {len(tags)} / speakers {len(clusters)}; "
            f"{len(tags_dec)} matched on the decoded derangement, {SELECT_N} selected",
            f"registered decoded-vs-deranged-decoded gap reproduced: {registered_gap:.6f} nats per "
            f"frame" + ("" if reference is None
                        else f"  (banked {float(reference['gap']):.6f}, "
                             f"|dev| {abs(float(reference['gap']) - registered_gap):.2e})"),
            f"mean log p: gold {lp_gold.mean():.2f}, deranged gold {lp_der.mean():.2f}, "
            f"decoded {lp_dec.mean():.2f}",
            # planner ruling 2026-09-22: the SIL-INCLUSIVE gold is the primary convention, and the
            # summary says so.  There is no second gold column here: this job scores ONE gold
            # string per utterance, and it is the SIL-inclusive one.
            "PRIMARY gold convention: SIL-INCLUSIVE -- the GoldPhonesJob string wrapped in one SIL "
            "at each end (s1a_job.with_edge_sil / _EDGE_SIL_NOTE), as the PER read's MFA gold and "
            "the training prior are; the decode is greedy_raw.json, run-collapsed argmax WITH its "
            "SIL tokens.  No SIL-stripped gold column is scored here.",
            "",
        ]
        for key, col in record["contrasts"].items():
            for scale in ("per_utterance", "per_gold_token", "per_frame"):
                c = col[scale]
                lines.append(f"{key:28s} {scale:15s} mean {c['mean']:+.6f} "
                             f"[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] median {c['median']:+.6f}")
        lines += [
            "",
            f"decoded scores above gold on {record['n_decoded_above_gold']} of {len(tags)} "
            f"utterances ({record['frac_decoded_above_gold']:.3f})",
            f"95 % CI: speaker-clustered bootstrap, {self.n_boot} resamples, seed {self.seed}",
            "",
        ]
        with open(self.out_report.get_path(), "w") as fh:
            fh.write("\n".join(lines))
        print("\n".join(lines), flush=True)


# =============================================================================================
# wiring: the source jobs' constructor arguments -> items -> RETURNN forward -> reader
# =============================================================================================
def _returnn_tools(returnn_exe, returnn_root):
    if returnn_exe is None or returnn_root is None:
        from ..default_tools import RETURNN_EXE, RETURNN_ROOT

        returnn_exe = RETURNN_EXE if returnn_exe is None else returnn_exe
        returnn_root = RETURNN_ROOT if returnn_root is None else returnn_root
    return returnn_exe, returnn_root


def derangement_gap(*, name: str, reverse_checkpoint: tk.Path, raw_hyps: tk.Path, gold: tk.Path,
                    units, eta_npz: tk.Path, split: str, reverse_kwargs=None,
                    returnn_exe: Optional[tk.Path] = None, returnn_root: Optional[tk.Path] = None):
    """The source ``BlankfreeDerangementGapJob(...)`` as items -> RFJv2 (CPU) -> reader.

    Returns ``{"items", "scores", "gap"}``; ``gap.out_summary`` is ``derangement_gap.json``.
    ``name`` is the alias stem (``.../{name}/...``).
    """
    returnn_exe, returnn_root = _returnn_tools(returnn_exe, returnn_root)
    items = ReverseGapItemsJob(raw_hyps=raw_hyps, gold=gold, units=units, eta_npz=eta_npz, split=split,
                               reverse_kwargs=reverse_kwargs, with_gold=False)
    items.add_alias(f"{_alias_prefix}/{name}/derangement_gap_items")
    scores = reverse_score_dump(name=f"{name}/derangement_gap", reverse_checkpoint=reverse_checkpoint,
                                items=items.out_items, returnn_exe=returnn_exe,
                                returnn_root=returnn_root, reverse_kwargs=reverse_kwargs)
    gap = BlankfreeDerangementGapJob(items=items.out_meta, scores=scores.out_files["scores.json"])
    gap.add_alias(f"{_alias_prefix}/{name}/derangement_gap")
    return {"items": items, "scores": scores, "gap": gap}


def decode_gap(*, name: str, reverse_checkpoint: tk.Path, raw_hyps: tk.Path, gold: tk.Path,
               units: Sequence[tk.Path], eta_npz: tk.Path, split: str,
               reference_gap: Optional[tk.Path] = None, reverse_kwargs=None,
               n_boot: int = PAIRED_PER_BOOT_RESAMPLES, seed: int = PAIRED_PER_BOOT_SEED,
               cluster: str = PAIRED_PER_CLUSTER, time_rqmt: float = 4.0,
               returnn_exe: Optional[tk.Path] = None, returnn_root: Optional[tk.Path] = None):
    """The source ``BlankfreeDecodeGapJob(...)`` as items -> RFJv2 (CPU) -> reader.

    ``name`` is both the alias stem and the free text the source stamped into the outputs;
    ``time_rqmt`` (source default 4.0 h, four scoring passes) goes to the forward job.  Returns
    ``{"items", "scores", "gap"}``.
    """
    returnn_exe, returnn_root = _returnn_tools(returnn_exe, returnn_root)
    items = ReverseGapItemsJob(raw_hyps=raw_hyps, gold=gold, units=units, eta_npz=eta_npz, split=split,
                               reverse_kwargs=reverse_kwargs, with_gold=True)
    items.add_alias(f"{_alias_prefix}/{name}/decode_gap_items")
    scores = reverse_score_dump(name=f"{name}/decode_gap", reverse_checkpoint=reverse_checkpoint,
                                items=items.out_items, returnn_exe=returnn_exe,
                                returnn_root=returnn_root, reverse_kwargs=reverse_kwargs,
                                time_rqmt=time_rqmt)
    gap = BlankfreeDecodeGapJob(items=items.out_meta, scores=scores.out_files["scores.json"],
                                reference_gap=reference_gap, name=name, n_boot=n_boot, seed=seed,
                                cluster=cluster)
    gap.add_alias(f"{_alias_prefix}/{name}/decode_gap")
    return {"items": items, "scores": scores, "gap": gap}
