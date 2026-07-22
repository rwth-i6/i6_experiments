"""
Sharded, full-node (4-GPU JUPITER) dev-phase scale tuning for the joint AED+CTC recog.

The non-sharded pipeline (:func:`aed_ctc_timesync_recog_recomb_auto_scale`) runs, on the dev set:

1. CTC time-sync recomb beam search (``model_recog_with_recomb``, ``aed_scale=0``) -> N-best,
2. AED rescore of that N-best (``aed_score`` -> ``aed_rescore_def``),
3. ``ScaleTuningJob`` over the two score sets (uses the ground-truth transcripts) -> optimal
   AED/CTC scales.

Steps 1+2 are GPU work. On JUPITER (4x GH200, flat-per-node billing, 12 h QOS wall) a single-GPU
search/rescore wastes 3/4 of the node, so we shard the dev set and run the shards across the whole
node via the generic :class:`forward_batched.BatchedReturnnForwardJob` engine (one work item per
shard; the per-shard outputs are merged by :class:`recog_batched.MergeSearchOutputShardsJob`).

The same checkpoint backs both the CTC search and the AED rescore, so each shard is ONE fused
forward config (checkpoint loaded once): the forward step runs the search, collapses the align-frame
hyps to label seqs (CTC collapse, in eager numpy -- there is no in-graph collapse op, and the
non-sharded path also collapses outside the search via ``ctc_alignment_to_label_seq``), then runs
the AED rescore on those label seqs, and writes two N-best score files: ``ctc.py.gz`` (label hyp +
CTC search score) and ``aed.py.gz`` (same label hyp + AED score).

The merged score files are post-processed (BPE->words etc) and fed to ``ScaleTuningJob``. This is the
testable dev phase; the combined first-pass search that consumes the tuned scales is built
separately (shard-based recog, reused from the fast-slow-RNA work).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sisyphus import tk


# Output filenames the fused forward callback writes into the work item's cwd.
_CTC_OUT_FILENAME = "ctc.py.gz"
_AED_OUT_FILENAME = "aed.py.gz"


def _fused_search_rescore_config(
    *,
    dataset,
    model_def,
    config: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
):
    """
    Build the fused per-shard forward config: same as a v3 search config, but with our fused
    forward step (search -> collapse -> AED rescore) and our two-file forward callback.
    """
    from i6_experiments.users.zeyer.recog import search_config_v3
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        model_recog_with_recomb,
    )

    config = dict(config or {})
    config["forward_step"] = _fused_search_rescore_forward_step
    config["forward_callback"] = _fused_get_forward_callback
    return search_config_v3(
        dataset=dataset,
        model_def=model_def,
        recog_def=model_recog_with_recomb,
        config=config,
        post_config=post_config,
    )


def _fused_search_rescore_forward_step(*, model, extern_data, **_kwargs_unused):
    """
    Run within RETURNN (eager forward). CTC search -> align->label collapse -> AED rescore.

    Marks three outputs: ``hyps`` (label seqs), ``ctc_scores``, ``aed_scores`` (each [batch, beam]).
    """
    import numpy as np
    import returnn.frontend as rf
    from returnn.tensor import Dim, batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        model_recog_with_recomb,
        aed_rescore_def,
    )

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()

    # CTC time-sync recomb beam search. aed_scale=0 -> pure CTC, hyps are align frames (with blank).
    hyps_wb, ctc_scores, out_spatial_dim, beam_dim = model_recog_with_recomb(
        model=model, data=data, data_spatial_dim=data_spatial_dim
    )
    blank_idx = model.blank_idx

    # Collapse align frames -> label seqs (eager numpy; no in-graph CTC collapse op available).
    hyps_raw = hyps_wb.copy_compatible_to_dims_raw([batch_dim, beam_dim, out_spatial_dim])  # [B, Beam, T]
    hyps_np = hyps_raw.detach().cpu().numpy()
    frame_lens = out_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim])  # [B]
    frame_lens_np = np.asarray(frame_lens.detach().cpu().numpy()).reshape(-1)
    n_batch, n_beam = hyps_np.shape[0], hyps_np.shape[1]
    if beam_dim.dyn_size_ext is not None:  # masked_select may make beam dynamic per batch
        beam_lens_np = np.asarray(
            beam_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim]).detach().cpu().numpy()
        ).reshape(-1)
    else:
        beam_lens_np = np.full((n_batch,), n_beam, dtype=np.int64)

    label_lists: List[List[List[int]]] = []
    max_len = 1
    for b in range(n_batch):
        per_beam = []
        for j in range(n_beam):
            if j >= beam_lens_np[b]:
                per_beam.append([])
                continue
            frames = hyps_np[b, j, : int(frame_lens_np[b])]
            labels_wb = _ctc_collapse_wb(frames, blank_idx)
            # wb id -> target (no-blank) id: blank removed from the vocab.
            labels = [int(x) if int(x) < blank_idx else int(x) - 1 for x in labels_wb]
            per_beam.append(labels)
            max_len = max(max_len, len(labels))
        label_lists.append(per_beam)

    labels_arr = np.zeros((n_batch, n_beam, max_len), dtype="int32")
    label_lens = np.zeros((n_batch, n_beam), dtype="int32")
    for b in range(n_batch):
        for j in range(n_beam):
            seq = label_lists[b][j]
            label_lens[b, j] = len(seq)
            if seq:
                labels_arr[b, j, : len(seq)] = seq

    dev = data.device
    label_lens_t = rf.convert_to_tensor(
        label_lens, dims=[batch_dim, beam_dim], dtype="int32", device=rf.get_default_dim_size_device()
    )
    label_spatial_dim = Dim(label_lens_t, name="aed_label_spatial")
    label_hyps = rf.convert_to_tensor(
        labels_arr, dims=[batch_dim, beam_dim, label_spatial_dim], dtype="int32", sparse_dim=model.target_dim, device=dev
    )

    # AED rescore on the collapsed label hyps (re-encodes data; same model/decoder).
    aed_scores = aed_rescore_def(
        model=model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        targets=label_hyps,
        targets_beam_dim=beam_dim,
        targets_spatial_dim=label_spatial_dim,
    )

    run_ctx = rf.get_run_ctx()
    run_ctx.mark_as_output(label_hyps, "hyps", dims=[batch_dim, beam_dim, label_spatial_dim])
    run_ctx.mark_as_output(ctc_scores, "ctc_scores", dims=[batch_dim, beam_dim])
    run_ctx.mark_as_output(aed_scores, "aed_scores", dims=[batch_dim, beam_dim])


def _ctc_collapse_wb(frames, blank_idx: int):
    """CTC collapse a per-frame best path (with-blank ids): drop blanks, merge repeats. -> list[int]."""
    import numpy as np

    frames = np.asarray(frames).reshape(-1)
    if frames.size == 0:
        return []
    change = np.empty(frames.shape, dtype=bool)
    change[0] = True
    change[1:] = frames[1:] != frames[:-1]
    keep = (frames != blank_idx) & change
    return frames[keep].tolist()


def _fused_get_forward_callback():
    """Forward callback writing two N-best score files: ctc.py.gz and aed.py.gz (same hyps)."""
    from typing import TextIO, Optional as _Optional
    from returnn.tensor import Tensor
    from returnn.forward_iface import ForwardCallbackIface

    class _FusedCallback(ForwardCallbackIface):
        def __init__(self):
            self.ctc_file: _Optional[TextIO] = None
            self.aed_file: _Optional[TextIO] = None

        def init(self, *, model):
            import gzip

            self.ctc_file = gzip.open(_CTC_OUT_FILENAME, "wt", encoding="utf-8")
            self.ctc_file.write("{\n")
            self.aed_file = gzip.open(_AED_OUT_FILENAME, "wt", encoding="utf-8")
            self.aed_file.write("{\n")

        def process_seq(self, *, seq_tag: str, outputs):
            hyps: Tensor = outputs["hyps"]  # [beam, label_spatial]
            ctc_scores: Tensor = outputs["ctc_scores"]  # [beam]
            aed_scores: Tensor = outputs["aed_scores"]  # [beam]
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # from the model target_dim
            hyps_len = hyps.dims[1].dyn_size_ext  # [beam] or []
            num_beam = hyps.raw_tensor.shape[0]

            self.ctc_file.write(f"{seq_tag!r}: [\n")
            self.aed_file.write(f"{seq_tag!r}: [\n")
            for i in range(num_beam):
                n = hyps_len.raw_tensor[i] if hyps_len.raw_tensor.shape else hyps_len.raw_tensor
                hyp_ids = hyps.raw_tensor[i, :n]
                hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
                self.ctc_file.write(f"  ({float(ctc_scores.raw_tensor[i])!r}, {hyp_serialized!r}),\n")
                self.aed_file.write(f"  ({float(aed_scores.raw_tensor[i])!r}, {hyp_serialized!r}),\n")
            self.ctc_file.write("],\n")
            self.aed_file.write("],\n")

        def finish(self):
            self.ctc_file.write("}\n")
            self.ctc_file.close()
            self.aed_file.write("}\n")
            self.aed_file.close()

    return _FusedCallback()


def aed_ctc_timesync_recog_recomb_auto_scale_batched(
    *,
    prefix: str,
    task,
    aed_ctc_model,
    aux_ctc_layer: Optional[int],
    num_shards: int,
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    recomb_type: str = "max",
    ctc_soft_collapse_threshold: Optional[float] = 0.8,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """
    Full sharded auto-scale pipeline, parallel over a full node. See module docstring.

    Dev phase: CTC N-best search + AED rescore -> ScaleTuningJob -> tuned AED scale (CTC fixed at 1.0).
    Then the first-pass joint AED+CTC recog (``model_recog_with_recomb`` with the tuned scale) over all
    ``task.eval_datasets``, sharded across the node via the same engine, scored to WER.

    :return: the first-pass :class:`ScoreResultCollection` (WERs on the eval sets).
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths
    from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob, _ShardedDataset
    from i6_experiments.users.zeyer.recog_batched import MergeSearchOutputShardsJob
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        get_aed_ctc_and_labelwise_prior,
    )
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput
    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    base_config: Dict[str, Any] = {
        "behavior_version": 24,
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "recog_recomb": recomb_type,
        "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
        "aux_loss_layers": [aux_ctc_layer] if aux_ctc_layer is not None else [],
        "beam_size": n_best_list_size,
    }
    if extra_config:
        from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep

        base_config = dict_update_deep(base_config, extra_config)

    # aed_scale=0.0 -> pure CTC search; the model still carries the AED decoder for the rescore.
    model = get_aed_ctc_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=0.0)
    dataset = task.dev_dataset

    # One fused (search + rescore) work item per shard, run across the node by the generic engine.
    work_items: Dict[str, Dict[str, Any]] = {}
    shard_keys: List[str] = []
    for s in range(num_shards):
        key = "shard_%03i" % s
        ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=s, seq_ordering="random")
        cfg = _fused_search_rescore_config(dataset=ds, model_def=model.definition, config=base_config)
        cfg = ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=model.checkpoint.path,
            returnn_config=cfg,
            log_verbosity=5,
            device="gpu",
        )
        work_items[key] = {
            "returnn_config": cfg,
            "model_checkpoint": model.checkpoint,
            "output_files": [_CTC_OUT_FILENAME, _AED_OUT_FILENAME],
        }
        shard_keys.append(key)

    job = BatchedReturnnForwardJob(
        work_items,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    job.add_alias(f"{prefix}/dev-search-rescore-batched")

    def merged(filename: str) -> tk.Path:
        """Single (num_shards=1) or merged (>1) output path for a given per-shard filename."""
        if num_shards == 1:
            return job.out_files[shard_keys[0]][filename]
        return MergeSearchOutputShardsJob(
            [job.out_files[k][filename] for k in shard_keys], output_gzip=filename.endswith(".gz")
        ).out_search_results

    ctc_scores = RecogOutput(output=merged(_CTC_OUT_FILENAME))
    aed_scores = RecogOutput(output=merged(_AED_OUT_FILENAME))

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE/SPM to words
        ctc_scores = f(ctc_scores)
        aed_scores = f(aed_scores)
        ref = f(ref)

    opt_scales_job = ScaleTuningJob(
        scores={"ctc": ctc_scores.output, "aed": aed_scores.output},
        ref=ref.output,
        fixed_scales={"ctc": 1.0},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    aed_scale = opt_scales_job.out_real_scale_per_name["aed"]

    # First-pass joint AED+CTC recog with the tuned scale (CTC fixed at 1.0), sharded over the node.
    first_pass_model = get_aed_ctc_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=aed_scale)
    recog_config = {
        **base_config,
        "beam_size": first_pass_recog_beam_size,
        # Beam-scaled batch size (encoder mem is ~beam-independent, but keep it simple/safe).
        "batch_size": int(
            20_000 * aed_ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
        ),
    }
    score = _combined_recog_batched(
        prefix=prefix, task=task, model=first_pass_model, config=recog_config, num_shards=num_shards
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", score.output)
    return score


def _combined_recog_batched(*, prefix: str, task, model, config: Dict[str, Any], num_shards: int):
    """
    First-pass joint AED+CTC recog over all ``task.eval_datasets``, sharded across the node.

    One ``model_recog_with_recomb`` search per (eval set x shard) as work items of the generic
    :class:`BatchedReturnnForwardJob`; the per-shard outputs are merged, post-processed (collapse
    blanks, BPE->words, take best), and scored. Mirrors the tail of the non-sharded
    ``aed_ctc_timesync_recog_recomb_auto_scale`` / ``recog_model``.

    :return: :class:`ScoreResultCollection` over the eval sets.
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from returnn_common.datasets_old_2022_10.interface import DatasetConfig
    from i6_experiments.users.zeyer import tools_paths
    from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob, _ShardedDataset
    from i6_experiments.users.zeyer.recog_batched import MergeSearchOutputShardsJob, _post_process_search_output
    from i6_experiments.users.zeyer.recog import (
        get_from_config,
        search_config_v2,
        search_config_v3,
        _v2_forward_out_filename,
        _v2_forward_ext_out_filename,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        model_recog_with_recomb,
    )

    recog_def = model_recog_with_recomb
    out_files = [_v2_forward_out_filename]
    if get_from_config((config, model.definition), "__recog_def_ext", False):
        out_files.append(_v2_forward_ext_out_filename)
    get_search_config = {None: search_config_v2, 1: search_config_v2, 2: search_config_v3}[
        get_from_config((config, model.definition), "__serialization_version", None)
    ]

    work_items: Dict[str, Dict[str, Any]] = {}
    cell_shard_keys: Dict[str, List[str]] = {}
    cell_datasets: Dict[str, DatasetConfig] = {}
    for ds_name, dataset in task.eval_datasets.items():
        assert "/" not in ds_name and "\\" not in ds_name, f"eval set name {ds_name!r} not usable as work-item key"
        shard_keys: List[str] = []
        for s in range(num_shards):
            if num_shards == 1:
                wkey = ds_name
                ds = dataset
            else:
                wkey = f"{ds_name}-sh{s:03}"
                ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=s, seq_ordering="sorted")
            cfg = get_search_config(dataset=ds, model_def=model.definition, recog_def=recog_def, config=config)
            cfg = ReturnnForwardJobV2.create_returnn_config(
                model_checkpoint=model.checkpoint, returnn_config=cfg, log_verbosity=5, device="gpu"
            )
            work_items[wkey] = {
                "returnn_config": cfg,
                "model_checkpoint": model.checkpoint,
                "output_files": list(out_files),
            }
            shard_keys.append(wkey)
        cell_shard_keys[ds_name] = shard_keys
        cell_datasets[ds_name] = dataset

    job = BatchedReturnnForwardJob(
        work_items,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    job.add_alias(f"{prefix}/recog-1stpass-batched")

    def cell_output(ds_name: str, filename: str):
        keys = cell_shard_keys[ds_name]
        if len(keys) == 1:
            return job.out_files[keys[0]][filename]
        return MergeSearchOutputShardsJob(
            [job.out_files[k][filename] for k in keys], output_gzip=filename.endswith(".gz")
        ).out_search_results

    outputs = {}
    for ds_name, dataset in cell_datasets.items():
        recog_out = _post_process_search_output(
            cell_output(ds_name, _v2_forward_out_filename),
            dataset=dataset,
            recog_def=recog_def,
            recog_post_proc_funcs=list(task.recog_post_proc_funcs),
        )
        outputs[ds_name] = task.score_recog_output_func(dataset, recog_out)
    return task.collect_score_results_func(outputs)
