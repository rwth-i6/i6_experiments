"""
Batched multi-GPU recog:
ONE SLURM Job covers all (epoch x test_set) recog cells of a training,
run once at the end, across the GPUs of one node
(JUPITER: 4 GH200, billing flat per node, must use all 4).

Replaces the per-(epoch, test_set) ``ReturnnForwardJobV2`` fan-out
used by ``recog_training_exp`` -> ``_RecogAndScoreFunc`` -> ``recog_model`` -> ``search_dataset``.

Design:
``recog_training_exp_batched`` builds one shared :class:`forward_batched.BatchedReturnnForwardJob`
whose work items are the full cross product ``model.fixed_epochs x task.eval_datasets``
(optionally each split into ``num_shards`` disjoint dataset shards for better GPU utilization /
to fit the 12h QOS wall on large corpora).
``fixed_epochs`` is known at graph-build time (the keep-epochs + last epoch),
so the whole item set is materialized upfront -- no per-epoch job fan-out.
The engine round-robins the items across the node's GPUs (one worker process per GPU),
is resumable (an item whose output exists is skipped), and barrier-stops before the walltime;
see :mod:`forward_batched` for the engine. When ``num_shards > 1`` the per-shard search outputs of
each cell are merged back (disjoint seq sets -> dict union) by :class:`_MergeSearchOutputShardsJob`.

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
import gzip
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union

from sisyphus import Job, Task, tk
from sisyphus import gs

from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchTakeBestJob
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.model_interfaces import RecogDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob, _ShardedDataset
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
    "recog_training_exp_batched",
]


class _MergeSearchOutputShardsJob(Job):
    """
    Merge the per-shard search outputs of one recog cell into a single search-output file.

    Each shard file is a gzipped python-literal dict ``{seq_tag: <value>}`` (value = the beam list
    for the main output, or list of dicts for the ext output) covering a disjoint set of seqs (the
    shards partition the dataset). Merge = dict union; any seq-tag overlap across shards is an error.
    """

    def __init__(self, shard_files: List[tk.Path], *, output_gzip: bool = True):
        super().__init__()
        assert shard_files, "_MergeSearchOutputShardsJob: no shard files to merge"
        self.shard_files = shard_files
        self.out_search_results = self.output_path("output.py.gz" if output_gzip else "output.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        merged: Dict[Any, Any] = {}
        for p in self.shard_files:
            path = p.get_path()
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                d = eval(f.read())  # {seq_tag: value}; our own trusted, repr-written data
            assert isinstance(d, dict), f"shard {path}: expected dict, got {type(d)}"
            overlap = merged.keys() & d.keys()
            assert not overlap, (
                f"shard {path}: {len(overlap)} seq tag(s) overlap across shards, e.g. {next(iter(overlap))!r}"
            )
            merged.update(d)

        out_path = self.out_search_results.get_path()
        opener = gzip.open if out_path.endswith(".gz") else open
        with opener(out_path, "wt") as f:
            f.write("{\n")
            for k, v in merged.items():
                f.write(f"{k!r}: {v!r},\n")
            f.write("}\n")


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

    Background: ``GetBestRecogTrainExp.update()`` is the sis graph-extension hook (runs in the manager,
    gated on ``scores_and_learning_rates.available()``, i.e. after training finishes). It calls
    ``get_relevant_epochs_from_training_learning_rate_scores`` to pick the best-by-train-score epochs
    (which can be kept epochs *not* in ``fixed_epochs``) and adds a recog for each.

    Here we build a single ``BatchedReturnnForwardJob`` over ``model.fixed_epochs`` upfront, and feed
    the pre-computed per-epoch scores via ``_PrecomputedRecogScore`` -- which only knows about
    ``fixed_epochs``. If ``update()`` asked it for a dynamically-picked non-fixed epoch it would
    ``KeyError``. So we disable ``update()``: we deliberately pick the best only among ``fixed_epochs``.
    The cost is small (the dynamic pick would, at most, find a slightly-better kept epoch near the end).

    --- Two ways to restore dynamic epoch picking, if ever worth the complexity (decided NOT worth it
    for now, 2026-06-01): ---

    Approach 1 (reuse scoring; recog stays one batched job):
    Write a ``GetBestRecogTrainExpBatched``. Override ``_add_recog`` to merely *collect* the epoch
    (no per-epoch job, no ``add_input``). Override ``update()`` so that once all epochs are gathered
    (fixed from ``__init__`` + dynamic from the base ``update`` logic), it builds ONE
    ``BatchedReturnnForwardJob`` over ``(all epochs x test_sets)``, then runs each cell through the
    existing ``_post_process_search_output`` + ``task.score_recog_output_func`` and ``add_input``s them.
    ``run()`` picks best as usual. Pro: reuses the post-proc + sclite sis jobs. The batched job is just
    constructed at update() time (post-training), which is the standard dynamic-extension pattern.

    Approach 2 (self-contained job owns the pipeline):
    A single job takes the model dir + the LR-scores file (training output) as inputs; at run time it
    calls ``get_relevant_epochs_from_training_learning_rate_scores`` itself, recogs all
    ``(epoch x test_set)`` cells across the 4 GPUs, and -- since sis ``output_path`` must be declared
    in ``__init__`` before the epoch set is known -- emits aggregate outputs only: one ``summary.json``
    (all results), a ``tk.Variable`` best-epoch, and a symlink to the best checkpoint
    (cf. ``GetBestPtCheckpointJob``). Downside: it must reimplement post-proc + scoring (take-best,
    spm->words, sclite via ``SCTK_PATH``, ref handling) internally. This monolithic shape is the right
    fit for the *auto-scale* recog (multi-phase search->rescore->scale-tune->search), so build that one
    this way; not worth forcing the plain recog into it.
    """

    def update(self):
        """Disabled: pick best only among model.fixed_epochs (see class docstring for why + alternatives)."""
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
    num_shards: int = 1,
    exclude_epochs: Collection[int] = (),
    model_avg: bool = False,
):
    """
    Cross-epoch batched recog, drop-in for ``recog_training_exp``.

    Builds ONE shared ``BatchedReturnnForwardJob`` whose work items are
    ``model.fixed_epochs x task.eval_datasets`` (each cell optionally split into ``num_shards``
    disjoint dataset shards), runs it once (at the end of training, when the checkpoints exist),
    then post-processes + scores each cell and picks the best epoch.
    Corpus-agnostic; one multi-GPU job instead of per-(epoch, test_set) fan-out.

    :param num_shards: if > 1, split each (epoch x test_set) cell into this many disjoint dataset
        shards (more, smaller work items -> better GPU utilization + fits the 12h wall on large
        eval sets). The per-shard search outputs are merged back per cell. Default 1 (one item per
        cell, unchanged).

    Same registered outputs as ``recog_training_exp``
    (``recog_results_best``, ``recog_results_all_epochs``).
    """
    assert getattr(model.definition, "backend", None) is not None, (
        "recog_training_exp_batched: TF (backend=None) path not supported; use recog_training_exp."
    )
    assert not model_avg, "recog_training_exp_batched: model_avg not supported yet (averaged ckpt not in cells)."
    assert num_shards >= 1

    epochs = sorted(e for e in model.fixed_epochs if e not in exclude_epochs)
    assert epochs, f"no fixed_epochs to recog (fixed_epochs={model.fixed_epochs}, exclude={exclude_epochs})"
    test_sets = task.eval_datasets  # name -> DatasetConfig

    out_files = [_v2_forward_out_filename]
    if get_from_config((search_config, model.definition), "__recog_def_ext", False):
        out_files.append(_v2_forward_ext_out_filename)
    get_search_config = {None: search_config_v2, 1: search_config_v2, 2: search_config_v3}[
        get_from_config((search_config, model.definition), "__serialization_version", None)
    ]

    # Build all work items: (epoch x test_set [x shard]), epoch-major.
    work_items: Dict[str, Dict[str, Any]] = {}
    cell_meta: Dict[str, Tuple[int, str, DatasetConfig]] = {}  # cell_key -> (epoch, dataset_name, dataset)
    cell_shard_keys: Dict[str, List[str]] = {}  # cell_key -> [work_item_key per shard]
    for epoch in epochs:
        model_with_checkpoint = model.get_epoch(epoch)
        for dataset_name, dataset in test_sets.items():
            assert isinstance(dataset_name, str) and isinstance(dataset, DatasetConfig)
            assert "/" not in dataset_name and "\\" not in dataset_name, (
                f"eval_set name {dataset_name!r} not usable as cell key (contains path separators)"
            )
            cell_key = f"ep{epoch:03}-{dataset_name}"
            shard_keys: List[str] = []
            for s in range(num_shards):
                if num_shards == 1:
                    work_item_key = cell_key
                    ds: DatasetConfig = dataset
                else:
                    work_item_key = f"{cell_key}-sh{s:03}"
                    ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=s, seq_ordering="sorted")
                returnn_config = get_search_config(
                    dataset=ds,
                    model_def=model.definition,
                    recog_def=recog_def,
                    config=search_config,
                    post_config=search_post_config,
                )
                # Transform the raw search config into a forward-ready one (task=forward + load=checkpoint),
                # exactly as ReturnnForwardJobV2 does -- otherwise rnn.py defaults to task=train.
                cell_config = ReturnnForwardJobV2.create_returnn_config(
                    model_checkpoint=model_with_checkpoint.checkpoint,
                    returnn_config=returnn_config,
                    log_verbosity=5,
                    device="gpu",
                )
                work_items[work_item_key] = {
                    "returnn_config": cell_config,
                    "model_checkpoint": model_with_checkpoint.checkpoint,
                    "output_files": list(out_files),
                }
                shard_keys.append(work_item_key)
            cell_shard_keys[cell_key] = shard_keys
            cell_meta[cell_key] = (epoch, dataset_name, dataset)

    batched_job = BatchedReturnnForwardJob(
        work_items,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    batched_job.add_alias(prefix_name + "/recog_batched")
    if gs.DEFAULT_ENVIRONMENT_SET.get("TMPDIR"):
        batched_job.set_env("TMPDIR", gs.DEFAULT_ENVIRONMENT_SET["TMPDIR"])

    def cell_output(cell_key: str, filename: str) -> tk.Path:
        """The single (num_shards=1) or merged (>1) output path of a cell for a given filename."""
        keys = cell_shard_keys[cell_key]
        if len(keys) == 1:
            return batched_job.out_files[keys[0]][filename]
        merge_job = _MergeSearchOutputShardsJob(
            [batched_job.out_files[k][filename] for k in keys],
            output_gzip=filename.endswith(".gz"),
        )
        merge_job.add_alias(prefix_name + f"/recog_batched_merge/{cell_key}")
        return merge_job.out_search_results

    # Per-cell post-processing + scoring (mirrors search_dataset's tail), collected per epoch.
    per_epoch_outputs: Dict[int, Dict[str, ScoreResultCollection]] = {}
    for cell_key, (epoch, dataset_name, dataset) in cell_meta.items():
        raw = cell_output(cell_key, _v2_forward_out_filename)
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
