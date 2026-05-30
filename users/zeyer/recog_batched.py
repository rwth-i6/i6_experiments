"""
Batched multi-GPU recog:
one SLURM Job covers all test_sets for a given epoch across the GPUs of one node
(JUPITER: 4 GH200, billing flat per node, must use all 4).
Drop-in alternative to the per-(epoch, test_set) ``ReturnnForwardJobV2`` fan-out
used by ``recog_training_exp`` -> ``_RecogAndScoreFunc`` -> ``recog_model`` -> ``search_dataset``.

v1: per-epoch batched.
One ``BatchedReturnnForwardJob`` per epoch covers all ``task.eval_datasets`` cells in one SLURM job.
Inside the job, a thread-pool of ``self.rqmt["gpu"]`` workers pinned to ``CUDA_VISIBLE_DEVICES=<i>``
pulls cells from a queue and runs ``rnn.py per_cell.config`` as a subprocess.
Per-cell outputs are exposed as a dict,
so the downstream post-processing + scoring sees them like a per-cell ``ReturnnForwardJobV2`` output.
Corpus-agnostic; no assumption about the test-set count.

Opt-in:
original ``recog_training_exp`` / ``_RecogAndScoreFunc`` / ``recog_model`` / ``search_dataset`` are unchanged;
the batched path is a parallel set of functions.
base-ls + other consumers keep their existing per-cell hashes by construction.

Cross-epoch batching (one job for ALL picked epochs across ALL test_sets) is a clean v2 --
it would create one ``BatchedReturnnForwardJob``
with a cell list materialized inside ``GetBestRecogTrainExp`` after picking.
Per-epoch v1 already gives the JUPITER wins
(4x GPU utilization, ~test_sets x within-epoch startup amortization);
v2 is a follow-up if startup still dominates.
"""

from __future__ import annotations

import functools
import json
import os
import queue as queue_mod
import subprocess
import threading
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union

from sisyphus import Job, Task, tk
from sisyphus import gs

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchTakeBestJob
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import RecogDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoint,
    ModelWithCheckpoints,
)
from i6_experiments.users.zeyer.recog import (
    GetBestRecogTrainExp,
    GetTorchAvgModelResult,
    RecogOutput,
    ScoreResultCollection,
    _RecogAndScoreFunc,
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
    "recog_model_batched",
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

            # Each cell runs rnn.py with cwd = outputs/<cell_key>/,
            # where its declared output_files land.
            out_dir = os.path.join("outputs", ck)
            os.makedirs(out_dir, exist_ok=True)

            manifest.append(
                {
                    "key": ck,
                    "config": os.path.abspath(config_path),
                    "model_checkpoint": cs["model_checkpoint"].get_path(),
                    "output_dir": os.path.abspath(out_dir),
                    "output_files": list(cs["output_files"]),
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
                try:
                    print(f"[BatchedRecog] GPU {gpu_idx}: starting cell {cell['key']}", flush=True)
                    subprocess.run([python, rnn_py, cell["config"]], cwd=cell["output_dir"], env=env, check=True)
                    print(f"[BatchedRecog] GPU {gpu_idx}: cell {cell['key']} done", flush=True)
                except BaseException as exc:
                    # Capture + continue so other cells finish.
                    with errors_lock:
                        errors.append((cell["key"], exc))
                    print(f"[BatchedRecog] GPU {gpu_idx}: cell {cell['key']} FAILED: {exc!r}", flush=True)
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


# -------------------------------------------------------------------------------------------------
# Integration:
# per-epoch batched search_dataset / recog_model / _RecogAndScoreFunc / recog_training_exp.
# -------------------------------------------------------------------------------------------------


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


def recog_model_batched(
    task: TaskCfg,
    model: ModelWithCheckpoint,
    recog_def: RecogDef,
    *,
    config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    recog_pre_post_proc_funcs_ext: Sequence[Callable] = (),
    eval_sets: Optional[Union[Collection[str], Dict[str, DatasetConfig]]] = None,
    dev_sets: Optional[Union[Collection[str], Dict[str, DatasetConfig]]] = None,
    name: Optional[str] = None,
) -> ScoreResultCollection:
    """
    Like ``recog.recog_model`` but bundles all ``eval_sets`` into ONE ``BatchedReturnnForwardJob``.

    Only the **inner Job creation** is replaced;
    epoch picking + scoring + result collection are unchanged.
    Backend selection: requires ``model.definition.backend`` (the torch path);
    the TF ``ReturnnSearchJobV2`` fallback is intentionally not supported here
    (new code, only the modern backend).
    """
    if dev_sets is not None:
        assert eval_sets is None, "cannot specify both eval_sets and dev_sets"
        eval_sets = dev_sets
    if eval_sets is not None:
        assert eval_sets
        if isinstance(eval_sets, dict):
            pass
        else:
            assert all(k in task.eval_datasets for k in eval_sets)
            eval_sets = {k: task.eval_datasets[k] for k in eval_sets}
    else:
        eval_sets = task.eval_datasets

    assert getattr(model.definition, "backend", None) is not None, (
        "recog_model_batched: TF (backend=None) path not supported; use recog_model."
    )

    # Build per-cell returnn_configs (one per test_set),
    # using the same builder as search_dataset.
    cells: Dict[str, Dict[str, Any]] = {}
    out_files = [_v2_forward_out_filename]
    if get_from_config((config, model), "__recog_def_ext", False):
        out_files.append(_v2_forward_ext_out_filename)
    get_search_config = {None: search_config_v2, 1: search_config_v2, 2: search_config_v3}[
        get_from_config((config, model), "__serialization_version", None)
    ]
    for dataset_name, dataset in eval_sets.items():
        assert isinstance(dataset_name, str) and isinstance(dataset, DatasetConfig)
        # Sanity: cell key is used as a path component.
        assert "/" not in dataset_name and "\\" not in dataset_name, (
            f"eval_set name {dataset_name!r} not usable as cell key (contains path separators)"
        )
        returnn_config = get_search_config(
            dataset=dataset,
            model_def=model.definition,
            recog_def=recog_def,
            config=config,
            post_config=search_post_config,
        )
        cells[dataset_name] = {
            "returnn_config": returnn_config,
            "model_checkpoint": model.checkpoint,
            "output_files": list(out_files),
        }

    batched_job = BatchedReturnnForwardJob(
        cells=cells,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    if name:
        batched_job.add_alias(name + "/recog_batched")
    if gs.DEFAULT_ENVIRONMENT_SET.get("TMPDIR"):
        batched_job.set_env("TMPDIR", gs.DEFAULT_ENVIRONMENT_SET["TMPDIR"])

    # Per-cell post-processing + scoring,
    # mirroring search_dataset's tail.
    outputs: Dict[str, ScoreResultCollection] = {}
    for dataset_name, dataset in eval_sets.items():
        raw = batched_job.out_files[dataset_name][_v2_forward_out_filename]
        recog_out = _post_process_search_output(
            raw,
            dataset=dataset,
            recog_def=recog_def,
            recog_post_proc_funcs=list(recog_post_proc_funcs) + list(task.recog_post_proc_funcs),
            recog_pre_post_proc_funcs_ext=recog_pre_post_proc_funcs_ext,
        )
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[dataset_name] = score_out
    return task.collect_score_results_func(outputs)


class _BatchedRecogAndScoreFunc(_RecogAndScoreFunc):
    """
    Same hash/contract as ``_RecogAndScoreFunc`` but per-epoch creates ONE ``BatchedReturnnForwardJob``.
    """

    def __call__(self, epoch_or_ckpt: Union[int, PtCheckpoint]) -> ScoreResultCollection:
        # Each call corresponds to a distinct (epoch / checkpoint) and thus a distinct BatchedReturnnForwardJob.
        # The alias must include that to avoid clobbering across epochs (sis warns + only the first sticks).
        if isinstance(epoch_or_ckpt, int):
            model_with_checkpoint = self.model.get_epoch(epoch_or_ckpt)
            alias_name = self.prefix_name + f"/epoch{epoch_or_ckpt:03}"
        elif isinstance(epoch_or_ckpt, PtCheckpoint):
            model_with_checkpoint = ModelWithCheckpoint(definition=self.model.definition, checkpoint=epoch_or_ckpt)
            # caller-provided checkpoint -- skip auto-alias; caller can alias the returned job if wanted
            alias_name = None
        else:
            raise TypeError(f"{self} unexpected type {type(epoch_or_ckpt)}")
        res = recog_model_batched(
            self.task,
            model_with_checkpoint,
            self.recog_def,
            config=self.search_config,
            search_post_config=self.search_post_config,
            recog_post_proc_funcs=self.recog_post_proc_funcs,
            name=alias_name,
        )
        if isinstance(epoch_or_ckpt, int):
            tk.register_output(self.prefix_name + f"/recog_results_per_epoch/{epoch_or_ckpt:03}", res.output)
        return res


def recog_training_exp_batched(
    prefix_name: str,
    task: TaskCfg,
    model: ModelWithCheckpoints,
    recog_def: RecogDef,
    *,
    search_config: Optional[Dict[str, Any]] = None,
    search_post_config: Optional[Dict[str, Any]] = None,
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
    search_mem_rqmt: Union[int, float] = 8,
    exclude_epochs: Collection[int] = (),
    model_avg: bool = False,
):
    """
    Drop-in for ``recog_training_exp`` that uses ``_BatchedRecogAndScoreFunc``:
    one multi-GPU job per epoch instead of one per (epoch, test_set).
    Same epoch picking (``GetBestRecogTrainExp``) and same output paths;
    only the inner fan-out changes.
    """
    recog_and_score_func = _BatchedRecogAndScoreFunc(
        prefix_name,
        task,
        model,
        recog_def,
        search_config=search_config,
        search_post_config=search_post_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
        search_mem_rqmt=search_mem_rqmt,
    )
    summarize_job = GetBestRecogTrainExp(
        exp=model,
        recog_and_score_func=recog_and_score_func,
        main_measure_lower_is_better=task.main_measure_type.lower_is_better,
        exclude_epochs=exclude_epochs,
    )
    summarize_job.add_alias(prefix_name + "/train-summarize")
    tk.register_output(prefix_name + "/recog_results_best", summarize_job.out_summary_json)
    tk.register_output(prefix_name + "/recog_results_all_epochs", summarize_job.out_results_all_epochs_json)
    if model_avg:
        model_avg_res_job = GetTorchAvgModelResult(
            exp=model, recog_and_score_func=recog_and_score_func, exclude_epochs=exclude_epochs
        )
        tk.register_output(prefix_name + "/recog_results_model_avg", model_avg_res_job.out_results)
