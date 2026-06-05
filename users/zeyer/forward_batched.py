"""
Batched multi-shard / multi-cell RETURNN forward on a full multi-GPU node (JUPITER policy).

:class:`BatchedReturnnForwardJob` is a generic engine: it runs a set of *work items*, each a
ready-to-run RETURNN forward config (``task=forward``, i.e. forced-align forward OR search/recog)
plus the output filename(s) that config writes, across the GPUs of one node and returns each item's
output(s) at a declared ``output_path``. The same engine drives:

- :func:`batched_forward_to_hdf` -- forward a dataset to HDF, split into ``num_shards >> num_gpus``
  disjoint logical shards (one HDF per shard), and
- ``recog_training_exp_batched`` (in ``recog_batched``) -- one job for all ``(epoch x test_set)``
  recog cells of a training, optionally each split into shards.

Why one node, sharded:
:func:`forward_to_hdf` runs ONE single-GPU :class:`ReturnnForwardJobV2` -> one HDF. On JUPITER
(4x GH200 / node, flat-per-node billing) that wastes 3/4 of the node, and for large corpora
(Loquacious ~25k h) a single GPU exceeds the 12 h QOS wall.

Engine properties:

- splits a dataset into disjoint logical shards via RETURNN's principled ``_num_shards`` /
  ``_shard_index`` (partition_epoch=1 -> disjoint partitions whose union is all seqs exactly once;
  see :func:`Dataset._apply_partition_epoch_and_sharding`) -- see :class:`_ShardedDataset`,
- runs ``num_gpus`` independent worker processes (no NCCL: forward is per-seq independent) that
  round-robin over the work items (rank r handles items r, r+G, r+2G, ...), each writing its
  item's declared output file(s),
- is resumable: an item whose output file(s) already exist is skipped, so a walltime kill just
  continues where it left off (the os.replace of each final file is atomic, so existence == done),
- proactively stops before the wall (mirrors :meth:`returnn...Engine._maybe_stop_for_resubmission`):
  when EMA-per-item-time * safety exceeds ``slurm_time_left_sec()``, a worker finishes its in-flight
  item, then waits at a filesystem barrier for the other still-active workers to reach their own
  stop point, so all exit together cleanly; the parent run() then exits as interrupted so sis
  resubmits via ``Task("run", resume="run")``,
- exposes ``completed_fraction`` = (#items fully written) / (#items) for sis ETA, mirroring
  :meth:`ReturnnTrainingJob.completed_fraction`.

Hashing: the job hashes on ``work_items`` (keys + each item's returnn_config + output filenames)
plus ``returnn_python_exe`` / ``returnn_root``. The node shape + stop policy live in ``self.rqmt``
and class attrs (not __init__ args -> not hashed).
"""

from __future__ import annotations

import functools

from typing import Any, Callable, Dict, List, Optional

from sisyphus import Job, Task, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ForwardRFDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
from i6_experiments.users.zeyer.forward_to_hdf import _returnn_forward_config_v2

__all__ = [
    "BatchedReturnnForwardJob",
    "batched_forward_to_hdf",
    "BatchedReturnnForwardDynamicJob",
    "batched_forward_per_arrow_shard_to_hdf",
]


class BatchedReturnnForwardJob(Job):
    """
    Run a set of RETURNN forward work items across a full multi-GPU node. See module docstring.

    Each work item is a ready-to-run RETURNN forward config (already passed through
    :meth:`ReturnnForwardJobV2.create_returnn_config`, so it has ``task=forward`` + the checkpoint
    load) together with the filename(s) it writes in its cwd. The engine runs them all, resumably,
    across the node's GPUs and places each item's output(s) at a declared ``output_path``.
    """

    # Runtime knobs (not __init__ args -> not hashed): node shape + stop policy.
    _num_gpus = 4  # JUPITER node = 4x GH200
    _stop_safety_factor = 1.2  # same default as returnn stop_for_resubmission_safety_factor
    _stop_exit_code = 3

    def __init__(
        self,
        work_items: Dict[str, Dict[str, Any]],
        *,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param work_items: mapping ``key -> {"returnn_config": ReturnnConfig,
            "output_files": list[str], "model_checkpoint": Optional[tk.Path]}``.
            ``key`` must be a valid path component (no ``/``). ``returnn_config`` must already be a
            forward-ready config (e.g. from :meth:`ReturnnForwardJobV2.create_returnn_config`); it is
            written to ``items/<key>/returnn.config`` and run with ``rnn.py`` in a fresh tmp cwd, so
            the file(s) named in ``output_files`` (written relative to cwd) are picked up there and
            moved to their declared ``output_path`` destinations. ``model_checkpoint`` is optional
            and only used for an existence pre-check (the checkpoint is already embedded in the
            config); the config is what is hashed.
        :param returnn_python_exe:
        :param returnn_root:
        """
        super().__init__()
        assert work_items, "BatchedReturnnForwardJob: work_items dict must be non-empty"
        for key, item in work_items.items():
            assert isinstance(key, str) and key and "/" not in key and "\\" not in key, (
                f"work item key {key!r} must be a non-empty path component"
            )
            assert isinstance(item.get("returnn_config"), ReturnnConfig), f"item {key}: returnn_config required"
            out_files = item.get("output_files")
            assert isinstance(out_files, (list, tuple)) and out_files, (
                f"item {key}: output_files must be a non-empty list"
            )
            for fn in out_files:
                assert isinstance(fn, str) and fn, f"item {key}: output_files entries must be non-empty str"

        self.work_items = work_items
        self.returnn_python_exe = (
            returnn_python_exe if returnn_python_exe is not None else tools_paths.get_returnn_python_exe()
        )
        self.returnn_root = returnn_root if returnn_root is not None else tools_paths.get_returnn_root()

        # Per-item outputs: dict key -> dict output_filename -> Path inside the Job's output/.
        import os

        self.out_files: Dict[str, Dict[str, tk.Path]] = {
            key: {fn: self.output_path(os.path.join("outputs", key, fn)) for fn in item["output_files"]}
            for key, item in work_items.items()
        }

        # Full node: 4 GPUs, all 288 cores, ~400 GiB host RAM; clipped to the 12 h QOS wall.
        self.rqmt = {
            "gpu": self._num_gpus,
            "gpu_mem": 96,
            "cpu": 72 * self._num_gpus,
            "mem": 100 * self._num_gpus,
            "time": 11.95,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_count(self) -> int:
        """:return: number of work items whose output file(s) are all already written."""
        import os

        n = 0
        for outs in self.out_files.values():
            if all(os.path.exists(p.get_path()) for p in outs.values()):
                n += 1
        return n

    def completed_fraction(self) -> float:
        """:return: fraction of items done (for sis progress/ETA). Mirrors ReturnnTrainingJob."""
        return self.completed_count() / len(self.work_items)

    def create_files(self):
        """Write one RETURNN forward config per work item + the worker manifest and driver."""
        import os

        items = []
        for key, item in self.work_items.items():
            item_dir = os.path.join("items", key)
            os.makedirs(item_dir, exist_ok=True)
            cfg_path = os.path.join(item_dir, "returnn.config")
            item["returnn_config"].write(cfg_path)
            items.append(
                {
                    "key": key,
                    "config": os.path.abspath(cfg_path),
                    # Where each file the config writes (relative to cwd) must end up.
                    "outputs": {fn: self.out_files[key][fn].get_path() for fn in item["output_files"]},
                }
            )
            ckpt = item.get("model_checkpoint")
            if ckpt is not None:
                ckpt_path = ckpt.get_path() if isinstance(ckpt, tk.Path) else getattr(ckpt, "path", ckpt).get_path()
                assert os.path.exists(ckpt_path), "item %s: checkpoint missing: %s" % (key, ckpt_path)

        _write_manifest(
            items=items,
            returnn_root_path=self.returnn_root.get_path(),
            returnn_python_exe_path=self.returnn_python_exe.get_path(),
        )
        _write_worker_driver()

    def run(self):
        """Spawn one worker per GPU; round-robin items; resubmit if stopped early for walltime."""
        codes = _spawn_and_wait_workers(
            num_gpus=self._num_gpus,
            returnn_python_exe_path=self.returnn_python_exe.get_path(),
            stop_safety_factor=self._stop_safety_factor,
        )
        _finish_run(
            codes=codes,
            n_done=self.completed_count(),
            n_total=len(self.work_items),
            stop_exit_code=self._stop_exit_code,
            label="BatchedReturnnForwardJob",
        )


def batched_forward_to_hdf(
    *,
    dataset: DatasetConfig,
    num_shards: int,
    model: Optional[ModelWithCheckpoint] = None,
    forward_def: Optional[ForwardRFDef] = None,
    forward_step: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    shard_seq_ordering: str = "random",
    alias_name: Optional[str] = None,
) -> List[tk.Path]:
    """
    Drop-in multi-GPU sharded counterpart of :func:`forward_to_hdf`, returning the list of shard
    HDFs (feed to ``HDFDataset(files=[...])``; no merge needed -- a MetaDataset matches seq tags
    across the shard HDFs).

    :param dataset: dataset to forward; its top-level ``get_main_dataset()`` dict must come from a
        dataset that ``supports_sharding()`` (e.g. OggZip). The shard params are injected at the top.
    :param num_shards: number of disjoint logical shards (>> num_gpus). Each becomes one output HDF.
    :param model: model whose checkpoint is loaded for the forward.
    :param forward_def: forward def (mark_as_output); see :func:`forward_to_hdf`.
    :param forward_step: alternatively a forward_step. Use either forward_def or forward_step.
    :param config: extra RETURNN config (e.g. ``model_outputs``, ``aux_loss_layers``, batch_size).
    :param forward_post_config: extra non-hashed RETURNN post config.
    :param shard_seq_ordering: seq ordering used when partitioning into shards. "random" gives
        statistically duration-balanced shards (partitions are index ranges over the ordered seqs);
        deterministic given epoch=1, so resume reproduces the same partition.
    :param alias_name:
    """
    assert not (forward_def and forward_step), "either forward_def or forward_step, not both"
    assert num_shards >= 1

    work_items: Dict[str, Dict[str, Any]] = {}
    keys: List[str] = []
    for i in range(num_shards):
        key = "shard_%03i" % i
        ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=i, seq_ordering=shard_seq_ordering)
        cfg = _returnn_forward_config_v2(
            dataset=ds,
            model_def=model.definition if model else None,
            forward_def=forward_def,
            forward_step=forward_step,
            config=config,
            post_config=forward_post_config,
        )
        cfg = ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=model.checkpoint.path if model else None,
            returnn_config=cfg,
            log_verbosity=5,
            device="gpu",
        )
        work_items[key] = {
            "returnn_config": cfg,
            "output_files": ["out.hdf"],
            "model_checkpoint": model.checkpoint if model else None,
        }
        keys.append(key)

    job = BatchedReturnnForwardJob(work_items)
    if alias_name:
        job.add_alias(alias_name)
    return [job.out_files[key]["out.hdf"] for key in keys]


class BatchedReturnnForwardDynamicJob(Job):
    """
    Like :class:`BatchedReturnnForwardJob`, but the set of work items (cells) is NOT known at
    __init__ -- it is enumerated at run time by a user callable. Hence a single directory output
    ``out/`` into which each cell writes its file(s) at a cell-chosen relative path, instead of one
    declared ``output_path`` per (statically known) cell. Use this when the cell list is only
    discoverable at run time (e.g. one cell per dynamically-discovered data shard); for a statically
    known cell set use :class:`BatchedReturnnForwardJob`.

    The cell enumeration separates cheap listing from expensive config building, so
    ``completed_fraction`` / resumability never build configs. ``enumerate_cells()`` returns an
    ordered ``list[(key, make_config, outputs)]`` where

    - ``key`` is a path component (no ``/``) naming the cell,
    - ``make_config`` is a zero-arg callable returning the cell's forward-ready ``ReturnnConfig``
      (already through :meth:`ReturnnForwardJobV2.create_returnn_config`); invoked only for cells
      whose output(s) are not yet present,
    - ``outputs`` maps each cwd filename the config writes to its destination path *relative to* this
      job's ``out/`` directory.

    ``enumerate_cells`` itself must be cheap + deterministic + picklable + hashable (e.g. a
    ``functools.partial`` of a module-level function); it is what the job hashes on (plus
    ``returnn_python_exe`` / ``returnn_root``). Same multi-GPU node engine + walltime barrier-stop +
    resumability (a cell whose output(s) exist is skipped) as :class:`BatchedReturnnForwardJob`.
    """

    # Runtime knobs (not __init__ args -> not hashed): node shape + stop policy.
    _num_gpus = 4
    _stop_safety_factor = 1.2
    _stop_exit_code = 3
    # FileCache tuning, injected into each cell's post_config in run() so it stays hash-free.
    # The 4 workers share one /tmp cache (fixed 96 GB tmpfs cap on the GH200 node). RETURNN's
    # proactive-cleanup "wanted" threshold defaults to 1 day, so within a multi-hour run no
    # released .arrow shard is ever old enough to evict and the cache grows until ENOSPC.
    # ~1 min lets the want-branch keep the 20% free margin during the run; it stays well above
    # the ~10 s in-use detection window, so shards being read are never evicted (each shard is
    # forwarded exactly once, so there is no re-copy cost to evicting promptly).
    _file_cache_opts = {"cleanup_files_wanted_older_than_days": 1.0 / (24 * 60)}

    def __init__(
        self,
        enumerate_cells: Callable[[], List[Any]],
        *,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param enumerate_cells: zero-arg callable returning ``list[(key, make_config, outputs)]``
            (see the class docstring). Invoked at run time (returnn on sys.path) and for
            ``completed_fraction``, so it must be cheap, deterministic, picklable + hashable.
        :param returnn_python_exe:
        :param returnn_root:
        """
        super().__init__()
        self.enumerate_cells = enumerate_cells
        self.returnn_python_exe = (
            returnn_python_exe if returnn_python_exe is not None else tools_paths.get_returnn_python_exe()
        )
        self.returnn_root = returnn_root if returnn_root is not None else tools_paths.get_returnn_root()

        # Single dynamic-count directory output: filled with each cell's file(s) in run().
        self.out_dir = self.output_path("out", directory=True)

        # Full node: 4 GPUs, all 288 cores, ~400 GiB host RAM; clipped to the 12 h QOS wall.
        self.rqmt = {
            "gpu": self._num_gpus,
            "gpu_mem": 96,
            "cpu": 72 * self._num_gpus,
            "mem": 100 * self._num_gpus,
            "time": 11.95,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _cell_dest(self, rel_dest: str) -> str:
        """:return: absolute destination of a cell output file (``rel_dest`` is under ``out/``)."""
        import os

        return os.path.join(self.out_dir.get_path(), rel_dest)

    def _cell_done(self, outputs: Dict[str, str]) -> bool:
        """:return: whether all of a cell's output files already exist."""
        import os

        return all(os.path.exists(self._cell_dest(rel)) for rel in outputs.values())

    def completed_count(self) -> int:
        """:return: number of cells whose output file(s) are all already written."""
        return sum(1 for (_key, _make_config, outputs) in self.enumerate_cells() if self._cell_done(outputs))

    def completed_fraction(self) -> float:
        """:return: fraction of cells done (for sis progress/ETA). Mirrors ReturnnTrainingJob."""
        cells = self.enumerate_cells()
        if not cells:
            return 0.0
        done = sum(1 for (_key, _make_config, outputs) in cells if self._cell_done(outputs))
        return done / len(cells)

    def create_files(self):
        """Write just the worker driver; the per-cell configs + manifest are built in run() (dynamic)."""
        _write_worker_driver()

    def run(self):
        """Enumerate cells, build a config per not-yet-done cell, run them all across the node."""
        import os
        import sys

        # Config building (whatever enumerate_cells' make_config does) may import returnn.
        sys.path.insert(0, self.returnn_root.get_path())
        os.makedirs(self.out_dir.get_path(), exist_ok=True)

        cells = self.enumerate_cells()
        assert cells, "enumerate_cells() returned no cells"

        items = []
        for key, make_config, outputs in cells:
            dests = {fn: self._cell_dest(rel) for fn, rel in outputs.items()}
            if all(os.path.exists(dest) for dest in dests.values()):
                continue  # resumable: this cell already done
            cfg = make_config()
            if self._file_cache_opts:
                cfg.post_config.setdefault("file_cache_opts", self._file_cache_opts)
            item_dir = os.path.join("items", key)
            os.makedirs(item_dir, exist_ok=True)
            cfg_path = os.path.join(item_dir, "returnn.config")
            cfg.write(cfg_path)
            items.append({"key": key, "config": os.path.abspath(cfg_path), "outputs": dests})

        if items:
            _write_manifest(
                items=items,
                returnn_root_path=self.returnn_root.get_path(),
                returnn_python_exe_path=self.returnn_python_exe.get_path(),
            )
            codes = _spawn_and_wait_workers(
                num_gpus=self._num_gpus,
                returnn_python_exe_path=self.returnn_python_exe.get_path(),
                stop_safety_factor=self._stop_safety_factor,
            )
        else:
            codes = [0]  # nothing left (all cells already present)

        _finish_run(
            codes=codes,
            n_done=self.completed_count(),
            n_total=len(cells),
            stop_exit_code=self._stop_exit_code,
            label="BatchedReturnnForwardDynamicJob",
        )


def batched_forward_per_arrow_shard_to_hdf(
    *,
    hf_data_dir: tk.Path,
    make_dataset: Callable[[List[str]], DatasetConfig],
    model: Optional[ModelWithCheckpoint] = None,
    forward_def: Optional[ForwardRFDef] = None,
    forward_step: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    forward_post_config: Optional[Dict[str, Any]] = None,
    alias_name: Optional[str] = None,
) -> tk.Path:
    """
    Co-sharded counterpart of :func:`batched_forward_to_hdf`, on :class:`BatchedReturnnForwardDynamicJob`:
    instead of ``num_shards`` seq-order partitions, produce exactly one HDF per ``data-*.arrow`` of
    ``hf_data_dir``, named by the arrow shard, returned as a single directory. A DistributeFilesDataset
    can then distribute file tuples ``(arrow_i, out_dir/<shard_key>.hdf)`` and pair audio with its
    alignment per subepoch.

    :param hf_data_dir: HF dataset dir holding the ``data-*.arrow`` shards.
    :param make_dataset: ``list[str] -> DatasetConfig`` building the forward dataset for one shard's
        arrow file(s). Must be picklable (e.g. a ``functools.partial`` of a module-level function).
    :param model: model whose checkpoint is loaded for the forward.
    :param forward_def: forward def (mark_as_output); use either this or forward_step.
    :param forward_step: alternatively a forward_step.
    :param config: extra RETURNN config.
    :param forward_post_config: extra non-hashed RETURNN post config.
    :param alias_name:
    :return: the ``out/`` directory Path holding ``<shard_key>.hdf`` per arrow shard.
    """
    assert not (forward_def and forward_step), "either forward_def or forward_step, not both"
    job = BatchedReturnnForwardDynamicJob(
        functools.partial(
            _enumerate_arrow_shard_cells,
            hf_data_dir=hf_data_dir,
            make_dataset=make_dataset,
            model=model,
            forward_def=forward_def,
            forward_step=forward_step,
            config=config,
            forward_post_config=forward_post_config,
        )
    )
    if alias_name:
        job.add_alias(alias_name)
    return job.out_dir


def _enumerate_arrow_shard_cells(
    *,
    hf_data_dir: tk.Path,
    make_dataset: Callable[[List[str]], DatasetConfig],
    model: Optional[ModelWithCheckpoint],
    forward_def: Optional[ForwardRFDef],
    forward_step: Optional[Callable],
    config: Optional[Dict[str, Any]],
    forward_post_config: Optional[Dict[str, Any]],
) -> List[Any]:
    """
    Enumerate one forward cell per ``data-*.arrow`` shard of ``hf_data_dir`` (cheap; no config built).

    Cell key = arrow basename without ``.arrow`` (e.g. ``data-00007-of-00848``); the config's
    ``out.hdf`` lands at ``<key>.hdf`` under the job's ``out/``. Uses the same
    :func:`get_arrow_shard_files_from_hf_dataset_dir` as the DFD pairing, so the sets line up.
    """
    import os
    from returnn.datasets.huggingface import get_arrow_shard_files_from_hf_dataset_dir

    cells = []
    for arrow_file in sorted(get_arrow_shard_files_from_hf_dataset_dir(hf_data_dir.get_path())):
        base = os.path.basename(arrow_file)
        key = base[: -len(".arrow")] if base.endswith(".arrow") else base
        make_config = functools.partial(
            _make_shard_forward_config,
            arrow_files=[arrow_file],
            make_dataset=make_dataset,
            model=model,
            forward_def=forward_def,
            forward_step=forward_step,
            config=config,
            forward_post_config=forward_post_config,
        )
        cells.append((key, make_config, {"out.hdf": key + ".hdf"}))
    return cells


def _make_shard_forward_config(
    *,
    arrow_files: List[str],
    make_dataset: Callable[[List[str]], DatasetConfig],
    model: Optional[ModelWithCheckpoint],
    forward_def: Optional[ForwardRFDef],
    forward_step: Optional[Callable],
    config: Optional[Dict[str, Any]],
    forward_post_config: Optional[Dict[str, Any]],
) -> ReturnnConfig:
    """Build the forward-ready ReturnnConfig for one shard (its arrow file(s)); called lazily in run()."""
    ds = make_dataset(arrow_files)
    cfg = _returnn_forward_config_v2(
        dataset=ds,
        model_def=model.definition if model else None,
        forward_def=forward_def,
        forward_step=forward_step,
        config=config,
        post_config=forward_post_config,
    )
    return ReturnnForwardJobV2.create_returnn_config(
        model_checkpoint=model.checkpoint.path if model else None,
        returnn_config=cfg,
        log_verbosity=5,
        device="gpu",
    )


class _ShardedDataset(DatasetConfig):
    """Wrap a DatasetConfig, injecting ``_num_shards`` / ``_shard_index`` into its main dataset dict."""

    def __init__(self, inner: DatasetConfig, *, num_shards: int, shard_index: int, seq_ordering: Optional[str] = None):
        super().__init__()
        self.inner = inner
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.seq_ordering = seq_ordering

    def get_main_dataset(self) -> Dict[str, Any]:
        """:return: inner main dataset dict + shard params at the top level."""
        d = dict(self.inner.get_main_dataset())
        d["_num_shards"] = self.num_shards
        d["_shard_index"] = self.shard_index
        if self.seq_ordering is not None:
            d["seq_ordering"] = self.seq_ordering
        return d

    def get_extern_data(self):
        return self.inner.get_extern_data()

    def get_default_input(self):
        return self.inner.get_default_input()

    def get_default_target(self):
        return self.inner.get_default_target()

    def get_main_name(self) -> str:
        return self.inner.get_main_name()


def _worker_main():
    """
    Entry point of the per-GPU worker (the generated ``worker.py`` imports and calls this).

    One worker process per GPU: round-robin over the manifest's work items
    (rank r handles items r, r+world, r+2*world, ...), run each item's config with rnn.py,
    skip items whose output file(s) already exist (resumable), and barrier-stop together with the
    other workers when low on walltime. Args ``--rank/--world/--manifest/--safety`` come from argv.
    """
    import argparse
    import json
    import os
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--safety", type=float, default=1.2)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    sys.path.insert(0, manifest["returnn_root"])

    from returnn.util.basic import slurm_time_left_sec

    items = manifest["items"]
    python_exe = manifest["python_exe"]
    rnn_py = manifest["rnn_py"]
    job_dir = os.path.dirname(os.path.abspath(args.manifest))

    if args.rank == 0:
        _log_node_usage("worker start")
    ema = None
    for si in range(args.rank, len(items), args.world):
        item = items[si]
        if _worker_item_done(item):  # atomic os.replace -> existence == complete
            continue
        if ema is not None:  # walltime-aware stop, mirrors returnn _maybe_stop_for_resubmission
            left = slurm_time_left_sec()
            if left is not None and left < ema * args.safety:
                _worker_barrier_and_exit(args.rank, args.world, job_dir, ema)
        if args.rank == 0:  # node-global /tmp+RAM snapshot, to diagnose the RAM-tmpfs ENOSPC
            _log_node_usage("before %s (item %i/%i)" % (item["key"], si, len(items)))
        t0 = time.time()
        _worker_run_item(python_exe, rnn_py, item)
        dt = time.time() - t0
        ema = dt if ema is None else 0.5 * ema + 0.5 * dt

    # Finished my stride: register so a stopping peer's barrier can complete, then exit clean.
    open(os.path.join(job_dir, "stopping.rank%i" % args.rank), "w").close()
    sys.exit(0)


def _worker_item_done(item):
    """:return: whether all of a work item's declared output files already exist."""
    import os

    return all(os.path.exists(dest) for dest in item["outputs"].values())


def _worker_run_item(python_exe, rnn_py, item):
    """Run one item's config with rnn.py in a fresh tmp cwd, then atomically move its outputs into place."""
    import os
    import shutil
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([python_exe, rnn_py, item["config"]], cwd=tmp)
        for fn, dest in item["outputs"].items():
            src = os.path.join(tmp, fn)
            assert os.path.exists(src), "item produced no %s: %s" % (fn, item["config"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            staged = dest + ".inprogress"  # same fs as final -> os.replace is atomic
            shutil.move(src, staged)
            os.replace(staged, dest)


def _worker_barrier_and_exit(rank, world, job_dir, ema):
    """
    Stop cleanly when low on walltime, in lockstep with the other workers.

    Register at the filesystem barrier, then wait for the other still-active workers to reach their
    own stop point (finish their in-flight item) so all exit together with no aborted partial item;
    bounded by a timeout. Exits with code 3, which must match
    :attr:`BatchedReturnnForwardJob._stop_exit_code` -- run() treats it as a clean low-walltime stop
    (resubmit), not a failure.
    """
    import glob
    import os
    import sys
    import time

    open(os.path.join(job_dir, "stopping.rank%i" % rank), "w").close()
    deadline = time.time() + max(600.0, 3.0 * (ema or 0.0))
    while time.time() < deadline:
        if len(glob.glob(os.path.join(job_dir, "stopping.rank*"))) >= world:
            break
        time.sleep(5.0)
    sys.exit(3)


def _log_node_usage(tag):
    """
    Log node /tmp (RAM-backed tmpfs) + RAM usage, to diagnose the tmpfs ENOSPC crashes.

    Node-global view, so call from rank 0 only. ``du`` of /tmp's top level reveals which entry eats
    the 96 GB rootfs, given our own output staging is only ~12 MB/shard.
    """
    import subprocess
    import sys

    print("=== node-usage [%s] ===" % tag, flush=True)
    for cmd in (["df", "-h", "/tmp"], ["free", "-g"], ["du", "-xh", "--max-depth=1", "/tmp"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            sys.stdout.write(r.stdout)
            if r.stderr:
                sys.stdout.write(r.stderr)
        except Exception as e:
            print("  %s failed: %r" % (" ".join(cmd), e))
    sys.stdout.flush()


def _write_manifest(*, items: List[Dict[str, Any]], returnn_root_path: str, returnn_python_exe_path: str):
    """Write the worker ``manifest.json`` (the work items + the returnn/python paths used to run them)."""
    import os
    import json

    manifest = {
        "rnn_py": os.path.join(returnn_root_path, "rnn.py"),
        "returnn_root": returnn_root_path,
        "python_exe": returnn_python_exe_path,
        "items": items,
    }
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _write_worker_driver():
    """
    Write the per-GPU worker driver ``worker.py``.

    Serialize an import of the worker entry point with i6_core's serialization_v2, so the logic lives
    as a real function in this module (:func:`_worker_main`) instead of a source-string blob. The
    generated worker.py is run by a plain Python (not via rnn.py). serialization_v2 resolves the direct
    i6_experiments import (and its sys.path); we add the remaining non-base sys.path entries (sisyphus,
    returnn, ...) explicitly, then append the call.
    """
    import sys
    from i6_core.serialization.serialization_v2 import serialize_config, get_base_sys_path_list

    extra_sys_paths = [p for p in sys.path if p not in get_base_sys_path_list()]
    worker_code = (
        serialize_config({"_worker_main": _worker_main}, extra_sys_paths=extra_sys_paths) + "\n_worker_main()\n"
    )
    with open("worker.py", "w") as f:
        f.write(worker_code)


def _spawn_and_wait_workers(*, num_gpus: int, returnn_python_exe_path: str, stop_safety_factor: float) -> List[int]:
    """Clear stale barrier markers, spawn one worker per GPU (round-robin items), wait; return exit codes."""
    import os
    import glob
    import subprocess

    # Clear stale barrier markers from a previous (interrupted) run.
    for f in glob.glob("stopping.rank*"):
        os.remove(f)

    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = parent_cvd.split(",") if parent_cvd else [str(i) for i in range(num_gpus)]
    world = len(gpu_ids)
    manifest = os.path.abspath("manifest.json")
    worker = os.path.abspath("worker.py")

    procs = []
    for r in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[r]
        cmd = [
            returnn_python_exe_path,
            worker,
            "--rank",
            str(r),
            "--world",
            str(world),
            "--manifest",
            manifest,
            "--safety",
            str(stop_safety_factor),
        ]
        procs.append(subprocess.Popen(cmd, env=env))
    return [p.wait() for p in procs]


def _finish_run(*, codes: List[int], n_done: int, n_total: int, stop_exit_code: int, label: str):
    """
    Shared post-spawn verdict for both engine jobs.

    All items written -> return (done). Else if every worker exited cleanly (0) or via the clean
    low-walltime stop (``stop_exit_code``), raise KeyboardInterrupt so sis resubmits ``run`` and the
    existing-output skip continues from the remaining items. Otherwise a real worker failure.
    """
    if n_done >= n_total:
        return
    if all(c in (0, stop_exit_code) for c in codes):
        raise KeyboardInterrupt("%s: stop for resubmission (%i/%i done)" % (label, n_done, n_total))
    raise RuntimeError("%s: worker failure codes=%s (%i/%i done)" % (label, codes, n_done, n_total))

