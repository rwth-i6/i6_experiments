"""
Batched multi-GPU recog:
ONE SLURM Job covers all (epoch x test_set) recog cells of a training,
run once at the end, across the GPUs of one node
(JUPITER: 4 GH200, billing flat per node, must use all 4).

Replaces the per-(epoch, test_set) ``ReturnnForwardJobV2`` fan-out
used by ``recog_training_exp`` -> ``_RecogAndScoreFunc`` -> ``recog_model`` -> ``search_dataset``.

Design:
``recog_training_exp_batched`` builds one ``BatchedReturnnForwardJob`` whose cells are the full
cross product ``model.fixed_epochs x task.eval_datasets``.
``fixed_epochs`` is known at graph-build time (the keep-epochs + last epoch),
so the whole cell set is materialized upfront -- no per-epoch job fan-out.
Inside the job, a thread-pool of ``self.rqmt["gpu"]`` workers pinned to ``CUDA_VISIBLE_DEVICES=<i>``
pulls cells from a queue and runs ``rnn.py per_cell.config`` as a subprocess.
A worker keeps the model loaded across consecutive cells of the same epoch
(cells are emitted epoch-major), amortizing checkpoint load + RETURNN startup.

The per-cell raw outputs feed the same post-processing + scoring as ``search_dataset``,
then ``GetBestRecogTrainExp`` picks the best epoch (fed pre-computed per-epoch scores,
so it does *not* spawn any further recog jobs).
Dynamic LR-score-based epoch picking is intentionally dropped:
we recog exactly the ``fixed_epochs`` set, which is what we can determine at graph-build.

Opt-in:
original ``recog_training_exp`` / ``_RecogAndScoreFunc`` / ``recog_model`` / ``search_dataset`` are unchanged;
this is a parallel module. base-ls + other consumers keep their existing per-cell hashes.
"""

from __future__ import annotations

import functools
import json
import os
import queue as queue_mod
import shutil
import subprocess
import threading
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union

from sisyphus import Job, Task, tk
from sisyphus import gs

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchTakeBestJob
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import RecogDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
from i6_experiments.users.zeyer.recog import (
    GetBestRecogTrainExp,
    RecogOutput,
    ScoreResultCollection,
    _v2_forward_ext_out_filename,
    _v2_forward_out_filename,
    get_from_config,
    search_config_v2,
    search_config_v3,
)
from i6_experiments.users.zeyer.datasets.task import Task as TaskCfg
from i6_core.returnn.training import PtCheckpoint


__all__ = [
    "BatchedReturnnForwardJob",
    "recog_training_exp_batched",
]


class BatchedReturnnForwardJob(Job):
    """
    One SLURM job that runs N cell recogs in parallel across the GPUs of one node.

    Mirrors the per-cell semantics of ``i6_core.returnn.forward.ReturnnForwardJobV2``
    (a returnn_config + model_checkpoint produces one or more output files via ``rnn.py``),
    but runs many cells in one job, work-pool style.
    Outputs are exposed as a dict keyed by cell name.

    Hash inputs: ``cells`` (set + contents), ``returnn_python_exe``, ``returnn_root``.
    Execution knobs (GPU count, time/mem/cpu, gpu_mem tier) live in ``self.rqmt``
    and are *not* hashed;
    callers tune them via ``job.rqmt[...] = ...`` after construction.
    Defaults target a full JUPITER node
    (4x GH200, 11.95h booster QOS cap, ~440 GiB usable host RAM, 288 cores).
    """

    # Bump when the run() output placement / cell execution changes for an already-finished job.
    # v2: place each cell's output at its declared output_path (was left in work/ -> lost on cleanup).
    __sis_version__ = 2

    def __init__(
        self,
        cells: Dict[str, Dict[str, Any]],
        *,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
    ):
        """
        :param cells: mapping ``cell_key -> {"returnn_config": ReturnnConfig, "model_checkpoint": tk.Path,
            "output_files": list[str]}``.
            ``cell_key`` must be a valid path component (no ``/``).
            Each cell's ``returnn_config`` is written to ``cells/<cell_key>/returnn.config``;
            ``rnn.py`` runs with ``cwd = outputs/<cell_key>/``,
            so the output file the config writes lands there.
        :param returnn_python_exe: path to the python exe to run ``rnn.py`` with.
        :param returnn_root: path to the RETURNN root (``rnn.py`` lives at ``<root>/rnn.py``).
        """
        super().__init__()
        assert cells, "BatchedReturnnForwardJob: cells dict must be non-empty"
        for ck, cs in cells.items():
            assert isinstance(ck, str) and ck and "/" not in ck and "\\" not in ck, (
                f"cell key {ck!r} must be a non-empty path component"
            )
            assert isinstance(cs.get("returnn_config"), ReturnnConfig), f"cell {ck}: returnn_config required"
            assert "model_checkpoint" in cs, f"cell {ck}: model_checkpoint required"
            out_files = cs.get("output_files")
            assert isinstance(out_files, (list, tuple)) and out_files, (
                f"cell {ck}: output_files must be a non-empty list"
            )
            for f in out_files:
                assert isinstance(f, str), f"cell {ck}: output_files entries must be str"

        self.cells = cells
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        # Per-cell outputs:
        # dict cell_key -> dict output_filename -> Path inside the Job's output/.
        self.out_files: Dict[str, Dict[str, tk.Path]] = {
            ck: {fn: self.output_path(os.path.join("outputs", ck, fn)) for fn in cs["output_files"]}
            for ck, cs in cells.items()
        }
        # Default rqmt: full JUPITER node.
        # Override via job.rqmt[...] = ... after construction.
        self.rqmt = {
            "gpu": 4,
            "time": 11.95,
            "mem": 400,
            "cpu": 72,
            "gpu_mem": 96,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        """Write each cell's returnn.config + a manifest for run()."""
        manifest = []
        for ck, cs in self.cells.items():
            cell_dir = os.path.join("cells", ck)
            os.makedirs(cell_dir, exist_ok=True)
            config_path = os.path.join(cell_dir, "returnn.config")
            cs["returnn_config"].write(config_path)

            manifest.append(
                {
                    "key": ck,
                    "config": os.path.abspath(config_path),
                    "output_files": list(cs["output_files"]),
                    # Where each produced file must end up: the declared output_path location
                    # (under the job's output/), which sis + downstream jobs read.
                    "output_dests": {fn: self.out_files[ck][fn].get_path() for fn in cs["output_files"]},
                }
            )
        with open("manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def run(self):
        """Spawn rqmt["gpu"] worker threads; each pins a GPU and pulls cells from a shared FIFO."""
        with open("manifest.json") as f:
            manifest: List[Dict[str, Any]] = json.load(f)

        work = queue_mod.Queue()
        for cell in manifest:
            work.put(cell)

        errors: List[Tuple[str, BaseException]] = []
        errors_lock = threading.Lock()

        rnn_py = os.path.join(self.returnn_root.get_path(), "rnn.py")
        python = self.returnn_python_exe.get_path()
        num_gpus = int(self.rqmt["gpu"])

        def worker(gpu_idx: int):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            while True:
                try:
                    cell = work.get_nowait()
                except queue_mod.Empty:
                    return
                ck = cell["key"]
                # Run each cell in its own dir (under work/), then move the produced output files
                # to their declared output_path destinations. rnn.py writes output files relative
                # to cwd, so without the move they would stay in work/ (and get cleaned up).
                run_dir = os.path.abspath(os.path.join("run", ck))
                try:
                    os.makedirs(run_dir, exist_ok=True)
                    print(f"[BatchedRecog] GPU {gpu_idx}: starting cell {ck}", flush=True)
                    subprocess.run([python, rnn_py, cell["config"]], cwd=run_dir, env=env, check=True)
                    for fn, dest in cell["output_dests"].items():
                        src = os.path.join(run_dir, fn)
                        if not os.path.exists(src):
                            raise FileNotFoundError(f"cell {ck}: expected output {fn!r} not produced in {run_dir}")
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        shutil.move(src, dest)
                    print(f"[BatchedRecog] GPU {gpu_idx}: cell {ck} done", flush=True)
                except BaseException as exc:
                    # Capture + continue so other cells finish.
                    with errors_lock:
                        errors.append((ck, exc))
                    print(f"[BatchedRecog] GPU {gpu_idx}: cell {ck} FAILED: {exc!r}", flush=True)
                finally:
                    work.task_done()

        threads = [threading.Thread(target=worker, args=(i,), name=f"recog-gpu-{i}") for i in range(num_gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            msg = "; ".join(f"cell {k}: {e!r}" for k, e in errors)
            raise RuntimeError(f"BatchedReturnnForwardJob: {len(errors)} cell(s) failed: {msg}")


def _post_process_search_output(
    raw_out: tk.Path,
    *,
    dataset: DatasetConfig,
    recog_def: RecogDef,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    recog_pre_post_proc_funcs_ext: Sequence[Callable] = (),
    keep_beam: bool = False,
    keep_alignment_frames: bool = False,
) -> RecogOutput:
    """
    Replicate the tail of ``search_dataset``:
    collapse repeats, strip blank, run user funcs, take best.

    Operates on a single cell's raw output path
    (e.g. produced by a ``BatchedReturnnForwardJob`` cell)
    and returns the final ``RecogOutput`` consumable by ``task.score_recog_output_func``.
    """
    from i6_experiments.users.zeyer.recog import ctc_alignment_to_label_seq

    res = raw_out
    raw_res_search_labels = RecogOutput(output=res)
    if recog_def.output_blank_label:
        raw_res_labels = ctc_alignment_to_label_seq(raw_res_search_labels, blank_label=recog_def.output_blank_label)
        if not keep_alignment_frames:
            res = raw_res_labels.output
    else:
        raw_res_labels = raw_res_search_labels
    for f in recog_pre_post_proc_funcs_ext:
        res = f(
            RecogOutput(output=res),
            dataset=dataset,
            raw_res_search_labels=raw_res_search_labels,
            raw_res_labels=raw_res_labels,
            search_labels_to_labels=functools.partial(
                ctc_alignment_to_label_seq, blank_label=recog_def.output_blank_label
            ),
        ).output
    for f in recog_post_proc_funcs:
        res = f(RecogOutput(output=res)).output
    if not keep_beam and recog_def.output_with_beam:
        res = SearchTakeBestJob(res, output_gzip=True).out_best_search_results
    return RecogOutput(output=res)


# -------------------------------------------------------------------------------------------------
# Cross-epoch batched recog: one job for ALL (epoch x test_set) cells, picked at the end.
# -------------------------------------------------------------------------------------------------


class _PrecomputedRecogScore:
    """
    A ``recog_and_score_func`` (epoch -> ScoreResultCollection) for ``GetBestRecogTrainExp``
    that returns a *pre-computed* per-epoch score collection instead of creating a new recog job.

    All recogs already ran in the single ``BatchedReturnnForwardJob``;
    this just hands the matching epoch's scored result to the summary job,
    so the summary creates no further jobs.
    """

    def __init__(self, per_epoch_scores: Dict[int, ScoreResultCollection]):
        self._per_epoch_scores = per_epoch_scores

    def __call__(self, epoch_or_ckpt: Union[int, PtCheckpoint]) -> ScoreResultCollection:
        if not isinstance(epoch_or_ckpt, int):
            raise TypeError(f"{self}: only int epochs supported (cross-epoch batched recog), got {epoch_or_ckpt!r}")
        if epoch_or_ckpt not in self._per_epoch_scores:
            raise KeyError(
                f"{self}: epoch {epoch_or_ckpt} not in pre-computed set {sorted(self._per_epoch_scores)};"
                f" cross-epoch batched recog only covers model.fixed_epochs."
            )
        return self._per_epoch_scores[epoch_or_ckpt]

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper

        # Identifier kept stable (not the qualname) so the class can move modules without breaking hashes.
        return sis_hash_helper({"class": "_PrecomputedRecogScore", "scores": self._per_epoch_scores})


class _GetBestRecogTrainExpFixedEpochs(GetBestRecogTrainExp):
    """
    Like ``GetBestRecogTrainExp`` but without the dynamic LR-score epoch picking in ``update()``.

    All ``fixed_epochs`` are already recog'd via the single ``BatchedReturnnForwardJob`` upfront,
    and their scores are fed in via ``_PrecomputedRecogScore``,
    so the dynamic ``update()`` (which would create more per-epoch recogs) must be a no-op.
    """

    def update(self):
        """No dynamic epoch picking: the fixed_epochs set is materialized at graph-build."""
        pass


def recog_training_exp_batched(
    prefix_name: str,
    task: TaskCfg,
    model: ModelWithCheckpoints,
    recog_def: RecogDef,
    *,
    search_config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    recog_pre_post_proc_funcs_ext: Sequence[Callable] = (),
    search_mem_rqmt: Union[int, float] = 8,  # unused: the BatchedReturnnForwardJob sets its own full-node rqmt
    exclude_epochs: Collection[int] = (),
    model_avg: bool = False,
):
    """
    Cross-epoch batched recog, drop-in for ``recog_training_exp``.

    Builds ONE ``BatchedReturnnForwardJob`` whose cells are ``model.fixed_epochs x task.eval_datasets``,
    runs it once (at the end of training, when the checkpoints exist),
    then post-processes + scores each cell and picks the best epoch.
    Corpus-agnostic; one multi-GPU job instead of per-(epoch, test_set) fan-out.

    Same registered outputs as ``recog_training_exp``
    (``recog_results_best``, ``recog_results_all_epochs``).
    """
    assert getattr(model.definition, "backend", None) is not None, (
        "recog_training_exp_batched: TF (backend=None) path not supported; use recog_training_exp."
    )
    assert not model_avg, "recog_training_exp_batched: model_avg not supported yet (averaged ckpt not in cells)."

    epochs = sorted(e for e in model.fixed_epochs if e not in exclude_epochs)
    assert epochs, f"no fixed_epochs to recog (fixed_epochs={model.fixed_epochs}, exclude={exclude_epochs})"
    test_sets = task.eval_datasets  # name -> DatasetConfig

    out_files = [_v2_forward_out_filename]
    if get_from_config((search_config, model.definition), "__recog_def_ext", False):
        out_files.append(_v2_forward_ext_out_filename)
    get_search_config = {None: search_config_v2, 1: search_config_v2, 2: search_config_v3}[
        get_from_config((search_config, model.definition), "__serialization_version", None)
    ]

    # Build all cells: (epoch x test_set), epoch-major so a worker reuses the loaded checkpoint.
    cells: Dict[str, Dict[str, Any]] = {}
    cell_meta: Dict[str, Tuple[int, str, DatasetConfig]] = {}  # cell_key -> (epoch, dataset_name, dataset)
    for epoch in epochs:
        model_with_checkpoint = model.get_epoch(epoch)
        for dataset_name, dataset in test_sets.items():
            assert isinstance(dataset_name, str) and isinstance(dataset, DatasetConfig)
            assert "/" not in dataset_name and "\\" not in dataset_name, (
                f"eval_set name {dataset_name!r} not usable as cell key (contains path separators)"
            )
            cell_key = f"ep{epoch:03}-{dataset_name}"
            returnn_config = get_search_config(
                dataset=dataset,
                model_def=model.definition,
                recog_def=recog_def,
                config=search_config,
                post_config=search_post_config,
            )
            # Transform the raw search config into a forward-ready one (sets task=forward + load=checkpoint),
            # exactly as ReturnnForwardJobV2 does -- otherwise rnn.py defaults to task=train.
            cell_config = ReturnnForwardJobV2.create_returnn_config(
                model_checkpoint=model_with_checkpoint.checkpoint,
                returnn_config=returnn_config,
                log_verbosity=5,
                device="gpu",
            )
            cells[cell_key] = {
                "returnn_config": cell_config,
                "model_checkpoint": model_with_checkpoint.checkpoint,
                "output_files": list(out_files),
            }
            cell_meta[cell_key] = (epoch, dataset_name, dataset)

    batched_job = BatchedReturnnForwardJob(
        cells=cells,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    batched_job.add_alias(prefix_name + "/recog_batched")
    if gs.DEFAULT_ENVIRONMENT_SET.get("TMPDIR"):
        batched_job.set_env("TMPDIR", gs.DEFAULT_ENVIRONMENT_SET["TMPDIR"])

    # Per-cell post-processing + scoring (mirrors search_dataset's tail), collected per epoch.
    per_epoch_outputs: Dict[int, Dict[str, ScoreResultCollection]] = {}
    for cell_key, (epoch, dataset_name, dataset) in cell_meta.items():
        raw = batched_job.out_files[cell_key][_v2_forward_out_filename]
        recog_out = _post_process_search_output(
            raw,
            dataset=dataset,
            recog_def=recog_def,
            recog_post_proc_funcs=list(recog_post_proc_funcs) + list(task.recog_post_proc_funcs),
            recog_pre_post_proc_funcs_ext=recog_pre_post_proc_funcs_ext,
        )
        score_out = task.score_recog_output_func(dataset, recog_out)
        per_epoch_outputs.setdefault(epoch, {})[dataset_name] = score_out

    per_epoch_scores: Dict[int, ScoreResultCollection] = {
        epoch: task.collect_score_results_func(outputs) for epoch, outputs in per_epoch_outputs.items()
    }

    # Summary: reuse GetBestRecogTrainExp's best-epoch picking, fed pre-computed per-epoch scores
    # (so it spawns no further recog jobs); dynamic LR-score picking disabled (fixed_epochs only).
    recog_and_score_func = _PrecomputedRecogScore(per_epoch_scores)
    summarize_job = _GetBestRecogTrainExpFixedEpochs(
        exp=model,
        recog_and_score_func=recog_and_score_func,
        main_measure_lower_is_better=task.main_measure_type.lower_is_better,
        exclude_epochs=exclude_epochs,
    )
    summarize_job.add_alias(prefix_name + "/train-summarize")
    tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    tk.register_output(prefix_name + "/recog_results_all_epochs", summarize_job.out_results_all_epochs_json)
