"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_reverse_init_jobs.py
(``_read_unit_hdfs``, ``load_seed_items_vad``) with ``s1a_job.with_edge_sil``.

ANALYSIS ONLY (uses transcripts): the seed's gold phone strings are the reverse model's targets.

:class:`SupervisedReverseDataJob` is the data half of the supervised reverse init, which the source
ran as one GPU job (``BlankfreeSupervisedReverseInitJob``) and this package runs as a RETURNN
training (``supervised.py``).  It loads the seed items exactly as the source did (every census and
cross-check of ``load_seed_items_vad``: 2849 seed IDs, the 2821 / 28 split, theta's CTC targets equal
the gold strings, non-empty SIL-free cores, the edge-SIL string feasible under ``ReverseConfig()``,
``z`` from the bed's rVAD-masked train unit HDFs), computes the source's batch inventory
(``supervised_steps.batch_inventory``: length buckets of 8, one ``torch.randperm`` per epoch from
``torch.Generator().manual_seed(42)``) and the held-out buckets (``length_buckets(held, 8)``), and
writes them as RETURNN data:

* ``train.hdf``: one sequence per (update, utterance) in the order the source visited them, i.e.
  all ``FIT.epochs`` epochs back to back (8 x 2821 sequences); sequence tag ``e{epoch}/{utt}``.
  The training reads it with ``seq_ordering="default"`` and ``partition_epoch = FIT.epochs``, so
  sub-epoch e is exactly the source's epoch e;
* ``held.hdf``: the 28 held-out utterances in held-bucket order;
* per sequence: ``data`` = the unit ids z (sparse, dim ``n_units``), ``phones`` = the edge-SIL
  gold string y (sparse, dim ``n_types``), ``eta`` = the speaker vector ([1, eta_dim] float32; the
  source's ``pack_batch`` casts it to float32 as well), ``batch_id`` = the update (train) or bucket
  (held) index, from which ``supervised_steps.SeedBatchIterDataPipe`` rebuilds the batches;
* ``schedule.json``: the counts and the per-update batch shapes, for the record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from sisyphus import Job, Task, tk

__all__ = ["SupervisedReverseDataJob", "load_seed_items_vad", "with_edge_sil", "write_seed_hdf"]

#: one SIL prepended and appended to every seed string (``s1a_job.EDGE_SIL``)
EDGE_SIL = True


def with_edge_sil(core: Sequence[int]) -> List[int]:
    """One SIL prepended and appended; identical for every condition."""
    from ..phones import SIL_ID

    return ([SIL_ID] + list(core) + [SIL_ID]) if EDGE_SIL else list(core)


def _read_unit_hdfs(paths: Sequence[str], wanted) -> dict:
    """``{tag: int64 units}`` for the ``wanted`` tags of RETURNN ``SimpleHDFWriter`` unit files."""
    import h5py

    wanted = set(wanted)
    rows = {}
    for path in paths:
        with h5py.File(path, "r") as fh:
            tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
            lengths = np.asarray(fh["seqLengths"])[:, 0].astype(np.int64)
            data = fh["inputs"][:]
            offset = 0
            for tag, length in zip(tags, lengths):
                if tag in wanted:
                    assert tag not in rows, f"duplicate unit tag {tag}"
                    rows[tag] = np.asarray(data[offset:offset + int(length)], dtype=np.int64).reshape(-1)
                offset += int(length)
            assert offset == len(data), (path, offset, len(data))
    return rows


def load_seed_items_vad(*, gold_json, ids_json, targets_hdf, train_segments, cv_segments,
                        units_hdfs, eta_npz):
    """The seed items ``(z, y, eta)`` with ``z`` from the bed's rVAD-masked unit HDFs.

    Every census and cross-check of the S2g loader is repeated unchanged (2849 seed IDs, 2821 / 28
    split, the theta CTC targets equal the gold strings, SIL-free non-empty cores, feasibility of
    the edge-SIL string under the UNCHANGED duration model).
    """
    import h5py

    from ..model.reverse import ReverseConfig, feasible
    from ..phones import PHONES

    gold = json.loads(Path(gold_json).read_text())
    ids = json.loads(Path(ids_json).read_text())
    train = Path(train_segments).read_text().splitlines()
    held = Path(cv_segments).read_text().splitlines()
    assert len(ids) == len(set(ids)) == 2849, "seed ID census differs from theta"
    assert len(train) == len(set(train)) == 2821 and len(held) == len(set(held)) == 28
    assert not set(train) & set(held), "seed train/CV overlap"
    assert set(train) | set(held) == set(ids) == set(gold), "seed gold/ID/split mismatch"
    phone2id = {p: i for i, p in enumerate(PHONES)}
    cores = {tag: [phone2id[p] for p in gold[tag]] for tag in ids}
    with h5py.File(targets_hdf, "r") as hdf:
        tags = [t.decode() if isinstance(t, bytes) else str(t) for t in hdf["seqTags"][:]]
        assert len(tags) == len(set(tags)) and set(tags) == set(ids), "theta target ID mismatch"
        offset = 0
        for tag, length in zip(tags, hdf["seqLengths"][:, 0]):
            target = hdf["inputs"][offset:offset + int(length)].reshape(-1)
            assert np.array_equal(target, np.asarray(cores[tag]) + 1), f"{tag}: theta CTC target mismatch"
            offset += int(length)
    units = _read_unit_hdfs(list(units_hdfs), ids)
    missing = sorted(set(ids) - set(units))
    assert not missing, f"{len(missing)} seed utterance(s) absent from the bed's unit HDFs, e.g. {missing[:3]}"
    with np.load(eta_npz, allow_pickle=False) as eta:
        positions = {str(t): i for i, t in enumerate(eta["tags"])}
        vectors = eta["eta"]
        cfg = ReverseConfig()
        items = {}
        for tag in train + held:
            z = units[tag]
            assert z.size and int(z.min()) >= 0 and int(z.max()) < cfg.n_units, f"{tag}: unit ids"
            y = with_edge_sil(cores[tag])
            assert cores[tag] and cfg.sil_id not in cores[tag], f"{tag}: expected nonempty SIL-free seed core"
            assert feasible(len(z), y, cfg), f"{tag}: infeasible unchanged duration model: S={len(z)}, U={len(y)}"
            items[tag] = (z, y, vectors[positions[tag]].copy())
    return train, [items[t] for t in train], held, [items[t] for t in held]


def write_seed_hdf(path: str, rows, *, n_batches: int) -> None:
    """Write ``rows = [(seq_tag, batch_id, (z, y, eta))]`` in order (module doc: the HDF layout)."""
    from returnn.datasets.hdf import SimpleHDFWriter

    from ..model.reverse import ReverseConfig

    cfg = ReverseConfig()
    writer = SimpleHDFWriter(
        path,
        dim=cfg.n_units,
        ndim=1,
        extra_type={"phones": (cfg.n_types, 1, "int32"), "eta": (cfg.eta_dim, 2, "float32"),
                    "batch_id": (int(n_batches), 1, "int32")},
    )
    for tag, bid, (z, y, eta) in rows:
        z = np.asarray(z, dtype=np.int32).reshape(1, -1)
        y = np.asarray(y, dtype=np.int32).reshape(1, -1)
        e = np.asarray(eta, dtype=np.float32).reshape(1, 1, cfg.eta_dim)
        writer.insert_batch(inputs=z, seq_len=[z.shape[1]], seq_tag=[tag],
                            extra={"phones": y, "eta": e, "batch_id": np.asarray([[bid]], dtype=np.int32)})
    writer.close()


def schedule_rows(tags, items, held_tags, held, cfg=None):
    """``(train_rows, n_updates, held_rows, n_held_batches, summary)`` in the source's visiting order."""
    from ..model.reverse import length_buckets
    from .supervised_steps import FIT, batch_inventory

    cfg = FIT if cfg is None else cfg
    batches, schedule = batch_inventory(items, cfg)
    train_rows = [(f"e{epoch}/{tags[i]}", update, items[i])
                  for update, (epoch, bi) in enumerate(schedule) for i in batches[bi]]
    held_batches = length_buckets(held, cfg.batch_size)
    held_rows = [(held_tags[i], hb, held[i]) for hb, idx in enumerate(held_batches) for i in idx]
    per_epoch = {e: sum(len(batches[bi]) for ep, bi in schedule if ep == e) for e in range(1, cfg.epochs + 1)}
    assert all(n == len(items) for n in per_epoch.values()), per_epoch
    summary = {
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "seed": cfg.seed,
        "train_utterances": len(items), "held_utterances": len(held),
        "batches_per_epoch": len(batches), "updates": len(schedule), "held_batches": len(held_batches),
        "train_sequences": len(train_rows),
        "schedule": [[epoch, bi, len(batches[bi])] for epoch, bi in schedule],
        "held_batch_sizes": [len(idx) for idx in held_batches],
    }
    return train_rows, len(schedule), held_rows, len(held_batches), summary


class SupervisedReverseDataJob(Job):
    """The supervised reverse init's train / held-out data in the source's batch order (module doc).

    ANALYSIS ONLY (uses transcripts).

    :param gold_json / ids_json / targets_hdf / train_segments / cv_segments: the seed inputs
        (``SeedGoldPhonesJob`` gold strings and ids, ``PhoneTargetHdfJob`` targets, the seed
        ``CvHoldoutSplitJob`` manifests).
    :param units_hdfs: the bed's rVAD-masked TRAIN unit HDFs (``units.train.shard*.hdf``).
    :param eta_npz: the bed's eta table.
    """

    def __init__(self, *, gold_json, ids_json, targets_hdf, train_segments, cv_segments,
                 units_hdfs: Sequence[tk.Path], eta_npz):
        super().__init__()
        self.inputs = dict(gold_json=gold_json, ids_json=ids_json, targets_hdf=targets_hdf,
                           train_segments=train_segments, cv_segments=cv_segments,
                           eta_npz=eta_npz)
        self.units_hdfs = list(units_hdfs)
        self.out_train_hdf = self.output_path("train.hdf")
        self.out_held_hdf = self.output_path("held.hdf")
        self.out_schedule = self.output_path("schedule.json")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        tags, items, held_tags, held = load_seed_items_vad(
            units_hdfs=[p.get_path() for p in self.units_hdfs],
            **{k: v.get_path() for k, v in self.inputs.items()})
        train_rows, n_updates, held_rows, n_held, summary = schedule_rows(tags, items, held_tags, held)
        write_seed_hdf(self.out_train_hdf.get_path(), train_rows, n_batches=n_updates)
        write_seed_hdf(self.out_held_hdf.get_path(), held_rows, n_batches=n_held)
        summary["train_frames_per_epoch"] = int(sum(len(z) for z, _, _ in items))
        summary["held_frames"] = int(sum(len(z) for z, _, _ in held))
        Path(self.out_schedule.get_path()).write_text(json.dumps(summary, indent=1) + "\n")
        print({k: v for k, v in summary.items() if k != "schedule"}, flush=True)
