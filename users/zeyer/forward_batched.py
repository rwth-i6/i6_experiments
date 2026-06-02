"""
Batched multi-shard forward-to-HDF on a full multi-GPU node (JUPITER policy).

:func:`forward_to_hdf` runs ONE single-GPU :class:`ReturnnForwardJobV2` -> one HDF. On JUPITER
(4x GH200 / node, flat-per-node billing) that wastes 3/4 of the node, and for large corpora
(Loquacious ~25k h) a single GPU exceeds the 12 h QOS wall. :class:`BatchedForwardJob` instead:

- splits the dataset into ``num_shards >> num_gpus`` disjoint logical shards via RETURNN's
  principled ``_num_shards`` / ``_shard_index`` (partition_epoch=1 -> disjoint partitions whose
  union is all seqs exactly once; see :func:`Dataset._apply_partition_epoch_and_sharding`),
- runs ``num_gpus`` independent worker processes (no NCCL: forward is per-seq independent) that
  round-robin over the shards (rank r handles shards r, r+G, r+2G, ...), each writing
  ``align.shard_{i}.hdf``,
- is resumable: a shard whose output HDF already exists is skipped, so a walltime kill just
  continues where it left off (the os.replace of the final HDF is atomic, so existence == done),
- proactively stops before the wall (mirrors :meth:`returnn...Engine._maybe_stop_for_resubmission`):
  when EMA-per-shard-time * safety exceeds ``slurm_time_left_sec()``, a worker finishes its
  in-flight shard, then waits at a filesystem barrier for the other still-active workers to reach
  their own stop point, so all exit together cleanly; the parent run() then exits as interrupted
  so sis resubmits via ``Task("run", resume="run")``,
- exposes ``completed_fraction`` = (#shard HDFs written) / num_shards for sis ETA, mirroring
  :meth:`ReturnnTrainingJob.completed_fraction`,
- outputs the shard HDFs as a file list (consume via ``HDFDataset(files=[...])``); no merge -- a
  :class:`MetaDataset` matches seq tags across the shard HDFs.

Same skeleton as the planned ``BatchedRecogJob``; the only variable is the per-shard work, so the
forward def/step is the parameter (swap for a recog callable to get batched recog).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from sisyphus import Job, Task, tk

from i6_core.returnn.forward import ReturnnForwardJobV2
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import ForwardRFDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
from i6_experiments.users.zeyer.forward_to_hdf import _returnn_forward_config_v2

__all__ = ["BatchedForwardJob", "batched_forward_to_hdf"]


class BatchedForwardJob(Job):
    """
    Forward a dataset to HDF, sharded across a full multi-GPU node. See module docstring.
    """

    # Runtime knobs (not __init__ args -> not hashed): node shape + stop policy.
    _num_gpus = 4  # JUPITER node = 4x GH200
    _stop_safety_factor = 1.2  # same default as returnn stop_for_resubmission_safety_factor
    _stop_exit_code = 3

    def __init__(
        self,
        *,
        dataset: DatasetConfig,
        num_shards: int,
        model: Optional[ModelWithCheckpoint] = None,
        forward_def: Optional[ForwardRFDef] = None,
        forward_step: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
        post_config: Optional[Dict[str, Any]] = None,
        shard_seq_ordering: str = "random",
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param dataset: dataset to forward; its top-level ``get_main_dataset()`` dict must come
            from a dataset that ``supports_sharding()`` (e.g. OggZip). The shard params are
            injected at the top level.
        :param num_shards: number of disjoint logical shards (>> num_gpus). Each becomes one
            output HDF.
        :param model: model whose checkpoint is loaded for the forward.
        :param forward_def: forward def (mark_as_output); see :func:`forward_to_hdf`.
        :param forward_step: alternatively a forward_step. Use either forward_def or forward_step.
        :param config: extra RETURNN config (e.g. ``model_outputs``, ``aux_loss_layers``, batch_size).
        :param post_config: extra non-hashed RETURNN post config.
        :param shard_seq_ordering: seq ordering used when partitioning into shards. "random" gives
            statistically duration-balanced shards (partitions are index ranges over the ordered
            seqs); deterministic given epoch=1, so resume reproduces the same partition.
        :param returnn_python_exe:
        :param returnn_root:
        """
        super().__init__()
        assert not (forward_def and forward_step), "either forward_def or forward_step, not both"
        assert num_shards >= 1
        self.dataset = dataset
        self.num_shards = num_shards
        self.model = model
        self.forward_def = forward_def
        self.forward_step = forward_step
        self.config = config
        self.post_config = post_config
        self.shard_seq_ordering = shard_seq_ordering
        self.returnn_python_exe = (
            returnn_python_exe if returnn_python_exe is not None else tools_paths.get_returnn_python_exe()
        )
        self.returnn_root = returnn_root if returnn_root is not None else tools_paths.get_returnn_root()

        self.out_hdf_files: List[tk.Path] = [self.output_path("align.shard_%03i.hdf" % i) for i in range(num_shards)]

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
        """:return: number of shards whose output HDF is already written."""
        import os

        return sum(os.path.exists(p.get_path()) for p in self.out_hdf_files)

    def completed_fraction(self) -> float:
        """:return: fraction of shards done (for sis progress/ETA). Mirrors ReturnnTrainingJob."""
        return self.completed_count() / self.num_shards

    def create_files(self):
        """Write one RETURNN forward config per shard + the worker manifest and driver."""
        import os
        import json

        os.makedirs("shards", exist_ok=True)
        shards = []
        for i in range(self.num_shards):
            cfg = self._shard_returnn_config(i)
            cfg_path = os.path.join("shards", "shard_%03i.config" % i)
            cfg.write(cfg_path)
            shards.append({"config": os.path.abspath(cfg_path), "hdf": self.out_hdf_files[i].get_path()})

        manifest = {
            "rnn_py": os.path.join(self.returnn_root.get_path(), "rnn.py"),
            "returnn_root": self.returnn_root.get_path(),
            "python_exe": self.returnn_python_exe.get_path(),
            "num_shards": self.num_shards,
            "shards": shards,
        }
        with open("manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open("worker.py", "w") as f:
            f.write(_WORKER_PY_SRC)

        if self.model is not None:
            ckpt = self.model.checkpoint.path.get_path()
            assert os.path.exists(ckpt), "checkpoint missing: %s" % ckpt

    def run(self):
        """Spawn one worker per GPU; round-robin shards; resubmit if stopped early for walltime."""
        import os
        import glob
        import subprocess

        # Clear stale barrier markers from a previous (interrupted) run.
        for f in glob.glob("stopping.rank*"):
            os.remove(f)

        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = parent_cvd.split(",") if parent_cvd else [str(i) for i in range(self._num_gpus)]
        world = len(gpu_ids)
        manifest = os.path.abspath("manifest.json")
        worker = os.path.abspath("worker.py")

        procs = []
        for r in range(world):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[r]
            cmd = [
                self.returnn_python_exe.get_path(),
                worker,
                "--rank",
                str(r),
                "--world",
                str(world),
                "--manifest",
                manifest,
                "--safety",
                str(self._stop_safety_factor),
            ]
            procs.append(subprocess.Popen(cmd, env=env))
        codes = [p.wait() for p in procs]

        n_done = self.completed_count()
        if n_done >= self.num_shards:
            return  # all shards written -> job done
        if all(c in (0, self._stop_exit_code) for c in codes):
            # Clean low-time stop (mirrors returnn Engine._maybe_stop_for_resubmission). Exit as
            # interrupted so sis resubmits via Task("run", resume="run"); the existing-HDF skip
            # makes the re-run continue from the remaining shards.
            # TODO confirm sis treats KeyboardInterrupt here as resubmit (not hard error) in this
            #   setup. Worst case the next walltime kill + resume + skip still makes progress, so
            #   the proactive stop is only an optimization, not required for correctness.
            raise KeyboardInterrupt("BatchedForwardJob: stop for resubmission (%i/%i done)" % (n_done, self.num_shards))
        raise RuntimeError("BatchedForwardJob: worker failure codes=%s (%i/%i done)" % (codes, n_done, self.num_shards))

    def _shard_returnn_config(self, shard_index: int):
        ds = _ShardedDataset(
            self.dataset,
            num_shards=self.num_shards,
            shard_index=shard_index,
            seq_ordering=self.shard_seq_ordering,
        )
        cfg = _returnn_forward_config_v2(
            dataset=ds,
            model_def=self.model.definition if self.model else None,
            forward_def=self.forward_def,
            forward_step=self.forward_step,
            config=self.config,
            post_config=self.post_config,
        )
        # Inject the checkpoint load + forward task defaults exactly like ReturnnForwardJobV2.
        return ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=self.model.checkpoint.path if self.model else None,
            returnn_config=cfg,
            log_verbosity=5,
            device="gpu",
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
    alias_name: Optional[str] = None,
) -> List[tk.Path]:
    """
    Drop-in multi-GPU sharded counterpart of :func:`forward_to_hdf`, returning the list of shard
    HDFs (feed to ``HDFDataset(files=[...])``; no merge needed).
    """
    job = BatchedForwardJob(
        dataset=dataset,
        num_shards=num_shards,
        model=model,
        forward_def=forward_def,
        forward_step=forward_step,
        config=config,
        post_config=forward_post_config,
    )
    if alias_name:
        job.add_alias(alias_name)
    return job.out_hdf_files


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


_WORKER_PY_SRC = r'''#!/usr/bin/env python3
"""Generated by BatchedForwardJob. One worker per GPU: round-robin over shards, write each shard
HDF atomically, skip already-written shards, barrier-stop together when low on walltime."""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_STOP_CODE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--safety", type=float, default=1.2)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    sys.path.insert(0, manifest["returnn_root"])
    try:
        from returnn.util.basic import slurm_time_left_sec
    except Exception:
        def slurm_time_left_sec():
            return None

    shards = manifest["shards"]
    python_exe = manifest["python_exe"]
    rnn_py = manifest["rnn_py"]
    job_dir = os.path.dirname(os.path.abspath(args.manifest))

    ema = None
    for si in range(args.rank, len(shards), args.world):
        sh = shards[si]
        if os.path.exists(sh["hdf"]):  # atomic os.replace -> existence == complete
            continue
        if ema is not None:  # walltime-aware stop, mirrors returnn _maybe_stop_for_resubmission
            left = slurm_time_left_sec()
            if left is not None and left < ema * args.safety:
                _barrier_and_exit(args.rank, args.world, job_dir, ema)
        t0 = time.time()
        _run_shard(python_exe, rnn_py, sh)
        dt = time.time() - t0
        ema = dt if ema is None else 0.5 * ema + 0.5 * dt

    # Finished my stride: register so a stopping peer's barrier can complete, then exit clean.
    open(os.path.join(job_dir, "stopping.rank%i" % args.rank), "w").close()
    sys.exit(0)


def _run_shard(python_exe, rnn_py, sh):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.check_call([python_exe, rnn_py, sh["config"]], cwd=tmp)
        out = os.path.join(tmp, "out.hdf")
        assert os.path.exists(out), "shard produced no out.hdf: %s" % sh["config"]
        staged = sh["hdf"] + ".inprogress"  # same fs as final -> os.replace is atomic
        shutil.move(out, staged)
        os.replace(staged, sh["hdf"])


def _barrier_and_exit(rank, world, job_dir, ema):
    # Wait for the other still-active workers to reach their own stop point (finish their in-flight
    # shard) so all exit together -- no aborted partial shard. Bounded by a timeout.
    open(os.path.join(job_dir, "stopping.rank%i" % rank), "w").close()
    deadline = time.time() + max(600.0, 3.0 * (ema or 0.0))
    while time.time() < deadline:
        if len(glob.glob(os.path.join(job_dir, "stopping.rank*"))) >= world:
            break
        time.sleep(5.0)
    sys.exit(_STOP_CODE)


if __name__ == "__main__":
    main()
'''
