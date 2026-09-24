"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/
config_sae_4a_supervised_blankfree_v1.py (``_schedules``, ``targets``, ``_train_config``, ``build``),
with ``sae/emc/blankfree_seed_jobs.py`` (``BlankfreeSeedSupportJob``), ``sae/emc/blankfree_seed.py``
(``seed_support_rows``), the dataset part of ``sae/emc/init_jobs.build_ctc_init_config`` and, for the
export, ``sae/emc/supervised_recognizer_export_jobs.py`` (replaced by ``ExtractSubmoduleCheckpointJob``).

ANALYSIS ONLY (uses transcripts): p0, a SUPERVISED blank-free recognizer trained on the labelled
10 h seed, selected on the held-out loss (itself a label-using selection).  Its checkpoint never
initialises a main-line arm.

WHAT IS IDENTICAL TO ``ctrl_20``: the model, the optimiser block, the gradient clip, the batch and
``emc_param_groups`` -- :func:`p0_train_config` starts from ``training.config.build_train_config``
with the bed's data dict, exactly as ``ctrl_20`` does.

WHAT DIFFERS FROM ``ctrl_20`` (each forced by "supervised, on the 10 h seed"):

1. THE DATA: the seed with its segment lists at partition epoch 1 (RETURNN filters a segment list
   after partitioning).  The targets HDF (the 2,849 seed utterances) is the MetaDataset's control
   dataset and the feature dump is read by tag (``HDFDataset.get_all_tags`` ignores
   ``seq_list_filter_file``, RETURNN #1504); the ordering key is therefore the collapsed TARGET
   length.
2. THE STREAMS: ``features`` + ``targets`` (the run-collapsed gold phone ids).
3. THE TRAIN STEP: ``reverse_model.p0_steps.train_step`` (``-log p`` of the gold string under the
   bed's blank-free topology; path-less utterances dropped and counted).
4. THE SCHEDULE: :data:`NUM_SUBEPOCHS` = 30 with ``budget_learning_rates(30, peak=1e-4)`` and the
   temperature pinned at the anneal's end value 2.0 (inert for this loss); kept checkpoints
   :data:`KEEP_EPOCHS` plus the best held-out one (``keep_best_n = 1``).

SELECTION: ``GetBestPtCheckpointJob`` on :data:`CV_LOSS_KEY`; the selected whole-model checkpoint's
``recognizer.`` submodule is sliced out by ``ExtractSubmoduleCheckpointJob`` into the recognizer-only
file that ``build_train_config(flat_checkpoint=...)`` loads.  (The banked selection is epoch 1.)

Port changes: the training job is a plain ``ReturnnTrainingJob`` (``training.jobs.train_arm``), which
resumes from its last checkpoint after an interrupted allocation where the source's bounded job ran
once without resume; the source's export job (a key / shape / dtype check of the slice against the
flat init) is replaced by ``ExtractSubmoduleCheckpointJob``; the dev-other PER reads of every kept
checkpoint (``blankfree_eval_jobs.epoch_reads``) are not built here (``analysis/``); the source
config's ``sys.path`` prolog is dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from sisyphus import Job, Task, tk

__all__ = [
    "NUM_SUBEPOCHS",
    "KEEP_EPOCHS",
    "KEEP_BEST_N",
    "TIME_RQMT",
    "CV_LOSS_KEY",
    "seed_support_rows",
    "BlankfreeSeedSupportJob",
    "p0_train_config",
    "p0_recognizer",
]

#: sub-epochs; a sub-epoch is ONE full pass over the 10 h seed (partition_epoch 1)
NUM_SUBEPOCHS = 30
#: kept checkpoints: every 5th sub-epoch (plus the best held-out one through ``keep_best_n``)
KEEP_EPOCHS = tuple(range(5, NUM_SUBEPOCHS + 1, 5))
#: ``cleanup_old_models``'s best-checkpoint count (post-config)
KEEP_BEST_N = 1
#: the wall clock of the single-GPU allocation
TIME_RQMT = 3.0
#: the held-out column the selection reads (``{dataset}_loss_{loss_name}``)
CV_LOSS_KEY = "dev_loss_blankfree_supervised"
#: the recognizer's submodule prefix inside a whole-model checkpoint
RECOGNIZER_PREFIX = "recognizer."

_PKG = __name__.rsplit(".", 2)[0]  # i6_experiments.users.wu.experiments.unsupervised_asr


# -------------------------------------------------------------------------------------------------
# the run-collapsed seed targets and the support census
# -------------------------------------------------------------------------------------------------


def seed_support_rows(ids, gold, train, held, retained, original, unit_lengths, phone_to_id,
                      *, sil_id=39, d_min=2, d_max=25, d_max_sil=50):
    from ..model.blankfree_seed import collapse_adjacent

    ids = list(ids)
    assert len(ids) == len(set(ids)) == 2849
    assert len(train) == len(set(train)) == 2821
    assert len(held) == len(set(held)) == 28
    train_set = set(train)
    assert not train_set & set(held)
    expected = set(ids)
    assert set(train) | set(held) == expected
    assert set(gold) == set(retained) == set(original) == set(unit_lengths) == expected
    rows = []
    for tag in ids:
        core = [phone_to_id[p] for p in gold[tag]]
        assert sil_id not in core, tag
        collapsed = collapse_adjacent(core)
        s = int(retained[tag])
        t = (s + 2) // 3
        phi = [sil_id] + core + [sil_id]
        phi_min = d_min * len(phi)
        phi_max = sum(d_max_sil if k == sil_id else d_max for k in phi)
        rows.append({
            "tag": tag, "split": "train" if tag in train_set else "held",
            "original_frames": int(original[tag]), "retained_frames": s,
            "unit_frames": int(unit_lengths[tag]), "output_frames": t,
            "original_tokens": len(core), "collapsed_tokens": len(collapsed),
            "merged": [[i, gold[tag][i]] for i in range(1, len(core)) if core[i] == core[i - 1]],
            "theta_supported": s > 0 and 0 < len(collapsed) <= t,
            "phi_min_frames": phi_min, "phi_max_frames": phi_max,
            "phi_supported": s > 0 and phi_min <= s <= phi_max and s == int(unit_lengths[tag]),
        })
    return rows


def _hdf_lengths(paths):
    import h5py

    lengths = {}
    for path in paths:
        with h5py.File(path.get_path(), "r") as hdf:
            for raw_tag, length in zip(hdf["seqTags"][:], hdf["seqLengths"][:, 0]):
                tag = raw_tag.decode() if isinstance(raw_tag, bytes) else str(raw_tag)
                assert tag not in lengths, tag
                lengths[tag] = int(length)
    return lengths


def _hdf_scalar_values(paths):
    import h5py

    values = {}
    for path in paths:
        with h5py.File(path.get_path(), "r") as hdf:
            offset = 0
            for raw_tag, length in zip(hdf["seqTags"][:], hdf["seqLengths"][:, 0]):
                tag = raw_tag.decode() if isinstance(raw_tag, bytes) else str(raw_tag)
                assert tag not in values and int(length) == 1, tag
                values[tag] = int(hdf["inputs"][offset])
                offset += 1
    return values


class BlankfreeSeedSupportJob(Job):
    """The run-collapsed 0-based gold targets of the seed (``collapsed_targets.hdf``, dim 40) and the
    support census (``support.json`` / ``support.txt``).  ANALYSIS ONLY (uses transcripts).

    The targets HDF of the theta CTC init (``old_targets_hdf``, ``phone id + 1``) is checked against
    the gold strings utterance by utterance.  ``features`` / ``units`` / ``originals`` are the bed's
    rVAD-masked TRAIN streams (feature / unit lengths and original frame counts for the census).
    """

    def __init__(self, *, gold_json, ids_json, old_targets_hdf, train_segments, cv_segments,
                 features, units, originals):
        super().__init__()
        self.gold_json, self.ids_json = gold_json, ids_json
        self.old_targets_hdf = old_targets_hdf
        self.train_segments, self.cv_segments = train_segments, cv_segments
        self.features, self.units, self.originals = list(features), list(units), list(originals)
        self.out_report = self.output_path("support.json")
        self.out_summary = self.output_path("support.txt")
        self.out_targets = self.output_path("collapsed_targets.hdf")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 0.5}, tries=1)

    def run(self):
        import h5py
        from returnn.datasets.hdf import SimpleHDFWriter

        from ..model.blankfree_seed import collapse_adjacent
        from ..phones import PHONES

        ids = json.loads(Path(self.ids_json.get_path()).read_text())
        gold = json.loads(Path(self.gold_json.get_path()).read_text())
        train = Path(self.train_segments.get_path()).read_text().splitlines()
        held = Path(self.cv_segments.get_path()).read_text().splitlines()
        retained_all, unit_all = _hdf_lengths(self.features), _hdf_lengths(self.units)
        original_all = _hdf_scalar_values(self.originals)
        retained = {tag: retained_all[tag] for tag in ids}
        units = {tag: unit_all[tag] for tag in ids}
        original = {tag: original_all[tag] for tag in ids}
        phone_to_id = {phone: i for i, phone in enumerate(PHONES)}
        rows = seed_support_rows(ids, gold, train, held, retained, original, units, phone_to_id)

        with h5py.File(self.old_targets_hdf.get_path(), "r") as hdf:
            tags = [tag.decode() if isinstance(tag, bytes) else str(tag) for tag in hdf["seqTags"][:]]
            assert len(tags) == len(set(tags)) and set(tags) == set(ids)
            offset = 0
            for tag, length in zip(tags, hdf["seqLengths"][:, 0]):
                expected = np.asarray([phone_to_id[p] + 1 for p in gold[tag]], dtype=np.int32)
                actual = np.asarray(hdf["inputs"][offset:offset + int(length)]).reshape(-1)
                assert np.array_equal(actual, expected), tag
                offset += int(length)

        counts = {}
        for split in ("train", "held"):
            group = [row for row in rows if row["split"] == split]
            counts[split] = {
                "ids": len(group), "original_tokens": sum(row["original_tokens"] for row in group),
                "collapsed_tokens": sum(row["collapsed_tokens"] for row in group),
                "merged_tokens": sum(len(row["merged"]) for row in group),
                "empty_retained": [row["tag"] for row in group if row["retained_frames"] == 0],
                "theta_unsupported": [row["tag"] for row in group if not row["theta_supported"]],
                "phi_unsupported": [row["tag"] for row in group if not row["phi_supported"]],
            }
        supported_all = all(not counts[split][key] for split in counts
                            for key in ("empty_retained", "theta_unsupported", "phi_unsupported"))
        Path(self.out_summary.get_path()).write_text(json.dumps({"supported_all": supported_all,
                                                                   "counts": counts}, indent=2) + "\n")
        writer = SimpleHDFWriter(self.out_targets.get_path(), dim=40, ndim=1)
        for tag in ids:
            target = np.asarray(collapse_adjacent([phone_to_id[p] for p in gold[tag]]), dtype=np.int32)
            writer.insert_batch(target[None, :], [len(target)], [tag])
        writer.close()
        Path(self.out_report.get_path()).write_text(json.dumps({"supported_all": supported_all,
                                                                  "counts": counts, "rows": rows}, indent=2) + "\n")


# -------------------------------------------------------------------------------------------------
# the training config
# -------------------------------------------------------------------------------------------------


def _schedules():
    """``(temperature_schedule, learning_rates)`` at :data:`NUM_SUBEPOCHS`: the budget round's own
    builders at their defaults (warmup 2 entries, hold to entry 18, linear decay to the floor), peak
    ``THETA_LEARNING_RATE``; the temperature pinned at the anneal's end value."""
    from ..training.config import THETA_LEARNING_RATE
    from ..training.schedules import budget_learning_rates, budget_temperature_schedule

    tau_end = budget_temperature_schedule(NUM_SUBEPOCHS)[-1]
    assert tau_end == 2.0, tau_end
    temperature_schedule = [tau_end] * NUM_SUBEPOCHS

    peak = THETA_LEARNING_RATE
    lr = budget_learning_rates(NUM_SUBEPOCHS, peak=peak)
    assert len(lr) == NUM_SUBEPOCHS, lr
    assert lr[0] < peak and lr[1] == peak, lr[:3]
    assert lr[17] == peak and lr[18] < peak, lr[16:20]
    assert lr[-1] == lr[0], (lr[0], lr[-1])
    return temperature_schedule, lr


def _seed_meta(feat_files, target_file, *, partition, ordering, segments) -> Dict[str, Any]:
    """``init_jobs.build_ctc_init_config``'s ``_meta`` in ``seq_order_control = "targets"`` mode."""
    from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
    from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

    assert segments is None or (partition or 1) == 1, (
        "a segment list needs partition_epoch 1; RETURNN filters after partitioning"
    )
    feats = HDFDataset(files=list(feat_files), partition_epoch=None, seq_ordering=None, segment_file=None)
    targets = HDFDataset(files=[target_file], partition_epoch=partition, seq_ordering=ordering,
                         segment_file=segments)
    return MetaDataset(
        data_map={"features": ("feats", "data"), "targets": ("targets", "data")},
        datasets={"feats": feats, "targets": targets},
        seq_order_control_dataset="targets",
    ).as_returnn_opts()


def p0_train_config(*, data: Dict[str, Any], targets_hdf: tk.Path, seed_train_segments: tk.Path,
                    seed_held_segments: tk.Path):
    """``ctrl_20``'s RETURNN config (``build_train_config(**data)`` at the 30-sub-epoch schedules)
    with the four documented deltas (module doc) and nothing else.

    :param data: the bed's ``build_train_config`` input dict (``training.arms`` convention); its
        train feature HDFs are the ones the seed features are read from.
    :param targets_hdf: ``BlankfreeSeedSupportJob.out_targets``.
    :param seed_train_segments: / :param seed_held_segments: the seed ``CvHoldoutSplitJob``'s
        2,821 / 28 manifests.
    """
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import
    from returnn.util.pprint import pformat

    from ..training.config import build_train_config
    from .p0_steps import TARGETS_KEY

    temperature_schedule, learning_rates = _schedules()
    cfg = build_train_config(**dict(data), num_subepochs=NUM_SUBEPOCHS,
                             temperature_schedule=temperature_schedule, learning_rates=learning_rates)

    # --- (1) + (2) the seed data and the streams the supervised step reads -----------------------
    features = data["train_feature_hdfs"]
    cfg.config["train"] = _seed_meta(features, targets_hdf, partition=1, ordering="laplace:.1000",
                                     segments=seed_train_segments)
    cfg.config["dev"] = _seed_meta(features, targets_hdf, partition=None, ordering="sorted",
                                   segments=seed_held_segments)
    cfg.config["newbob_multi_num_epochs"] = 1

    holders = [o for o in (cfg.python_prolog or []) + (cfg.python_epilog or []) if isinstance(o, Collection)]
    assert len(holders) == 1, f"expected one serializer Collection, got {len(holders)}"
    objects = holders[0].serializer_objects
    codes = [o for o in objects if isinstance(getattr(o, "code", None), str) and o.code.startswith("extern_data = ")]
    assert len(codes) == 1, f"expected one extern_data block, got {len(codes)}"
    extern_data = {
        "features": {"dim": 1024, "shape": (None, 1024), "dtype": "float16"},
        TARGETS_KEY: {"dim": 40, "shape": (None,), "sparse": True, "dtype": "int32"},
    }
    codes[0].code = f"extern_data = {pformat(extern_data)}\n"

    # --- (3) the supervised train step ----------------------------------------------------------
    steps = [o for o in objects if getattr(o, "import_as", None) == "train_step"]
    assert len(steps) == 1, f"expected one train_step import, got {len(steps)}"
    objects[objects.index(steps[0])] = Import(
        code_object_path=f"{_PKG}.reverse_model.p0_steps.train_step",
        unhashed_package_root=f"{_PKG}.reverse_model", import_as="train_step")

    # post-config (unhashed): keep the best held-out checkpoint on disk for the selection
    cleanup = cfg.post_config["cleanup_old_models"]
    assert cleanup["keep_best_n"] == 0, cleanup
    cleanup["keep_best_n"] = KEEP_BEST_N
    return cfg


def p0_recognizer(
    *,
    data: Dict[str, Any],
    gold_json: tk.Path,
    ids_json: tk.Path,
    old_targets_hdf: tk.Path,
    seed_train_segments: tk.Path,
    seed_held_segments: tk.Path,
    train_units_hdfs: Sequence[tk.Path],
    alias: Optional[str] = "sae/4a/supervised_blankfree",
) -> Dict[str, Any]:
    """p0 (ANALYSIS ONLY, uses transcripts): support job, training, best-held-out selection, and the
    recognizer-only export.

    ``data`` is the bed's ``build_train_config`` dict (its ``train_feature_hdfs`` /
    ``train_original_hdfs`` are the bed's rVAD-masked train streams); ``train_units_hdfs`` the
    matching unit HDFs (the census's unit lengths).  Returns ``support`` / ``train_config`` /
    ``train`` / ``best`` / ``export`` and ``checkpoint`` (the export's ``out_checkpoint``, the value
    for ``flat_checkpoint``).
    """
    from i6_core.returnn.training import GetBestPtCheckpointJob

    from ..training.checkpoints import ExtractSubmoduleCheckpointJob
    from ..training.jobs import train_arm

    out: Dict[str, Any] = {}
    support = BlankfreeSeedSupportJob(
        gold_json=gold_json, ids_json=ids_json, old_targets_hdf=old_targets_hdf,
        train_segments=seed_train_segments, cv_segments=seed_held_segments,
        features=data["train_feature_hdfs"], units=train_units_hdfs, originals=data["train_original_hdfs"],
    )
    out["support"] = support
    cfg = p0_train_config(data=data, targets_hdf=support.out_targets,
                          seed_train_segments=seed_train_segments, seed_held_segments=seed_held_segments)
    out["train_config"] = cfg
    train = train_arm("training", cfg, NUM_SUBEPOCHS, keep_epochs=KEEP_EPOCHS, time_rqmt=TIME_RQMT,
                      alias_prefix=alias)
    out["train"] = train
    best = GetBestPtCheckpointJob(model_dir=train.out_model_dir, learning_rates=train.out_learning_rates,
                                  key=CV_LOSS_KEY, index=0)
    out["best"] = best
    export = ExtractSubmoduleCheckpointJob(checkpoint=best.out_checkpoint.path, prefix=RECOGNIZER_PREFIX)
    out["export"] = export
    out["checkpoint"] = export.out_checkpoint
    if alias:
        support.add_alias(f"{alias}/support")
        best.add_alias(f"{alias}/best")
        export.add_alias(f"{alias}/recognizer_export")
    return out
