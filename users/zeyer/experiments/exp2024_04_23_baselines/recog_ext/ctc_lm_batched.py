"""
Batched (full-node, sharded) CTC+LM recog with labelwise prior and auto scale tuning.

FZJ variant of :func:`ctc_recog_ext.ctc_recog_recomb_labelwise_prior_auto_scale`:
every GPU stage runs as work items of one :class:`BatchedReturnnForwardJob`
(FZJ bills per node, so single-GPU jobs are not allowed there),
CPU stages (prior renorm, prior scoring, scale tuning, scoring) stay as-is.

Stage mapping:

- labelwise prior: :func:`get_ctc_prior_probs_sharded` (N shards over the train data on one node,
  exact weighted merge via the per-shard frame counts) instead of one single-GPU statistics forward.
- dev N-best: one fused work item per shard runs the CTC recomb search AND scores the N-best
  with the attached LM (``model.lm``), replacing the separate single-GPU LM rescore job.
  The LM is detached during the search itself, so the search is pure CTC
  (same N-best as the non-batched CTC-only search; the prior is scored separately, also as there).
- first pass: the shared :func:`aed_ctc_batched._combined_recog_batched`, with the CTC+LM recog def
  and the LM+prior-wrapped model.

The intermediate rescore-with-optimal-scales step of the non-batched pipeline is left out
(it would need its own GPU rescore jobs and the first pass is what we report).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sisyphus import Job, Task, tk


# Output filenames the fused forward callback writes into the work item's cwd.
_AM_OUT_FILENAME = "am.py.gz"
_LM_OUT_FILENAME = "lm.py.gz"


def ctc_recog_recomb_labelwise_prior_auto_scale_batched(
    *,
    prefix: str,
    task,
    ctc_model,
    lm,
    aux_ctc_layer: Optional[int],
    num_shards: int,
    use_prior: bool = True,
    labelwise_prior=None,
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    recomb_type: str = "max",
    ctc_soft_collapse_threshold: Optional[float] = 0.8,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """
    Full sharded CTC+LM auto-scale pipeline, parallel over a full node. See module docstring.

    Dev phase: pure-CTC N-best search + fused LM scoring (+ separate prior scoring)
    -> ScaleTuningJob {am fixed 1, prior negative relative to lm, lm}.
    Then the first-pass CTC+LM+prior recog (``model_recog_with_recomb`` with the tuned scales)
    over all ``task.eval_datasets``, sharded across the node, scored to WER.

    :param ctc_model: e.g. an AED+CTC model; the CTC comes from aux layer ``aux_ctc_layer``
        (via ``enc_aux_logits[-1]`` in ``encode_and_get_ctc_log_probs``)
    :param lm: e.g. from ``ctc_recog_ext._get_lm_model``
    :param labelwise_prior: e.g. a :class:`Prior` from the train transcription label counts
        (:func:`collect_model_dataset_stats.compute_label_prior_log_probs`, CPU-only,
        about as good as the model softmax prior). If None and ``use_prior``,
        falls back to the model softmax prior via :func:`get_ctc_prior_probs_sharded` (GPU).
    :return: the first-pass :class:`ScoreResultCollection` (WERs on the eval sets)
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths
    from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob, _ShardedDataset
    from i6_experiments.users.zeyer.recog_batched import MergeSearchOutputShardsJob
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput
    from i6_experiments.users.zeyer.datasets.utils.vocab import get_vocab_file_from_task, ExtendVocabLabelsByNewLabelJob
    from i6_experiments.users.zeyer.decoding.lm_rescoring import prior_score
    from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior, PriorRemoveLabelRenormJob
    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        get_ctc_with_lm_and_labelwise_prior,
        model_recog as ctc_ext_model_recog,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import _ctc_model_def_blank_idx
    from .aed_ctc_batched import _combined_recog_batched
    from .ctc import model_recog_with_recomb

    vocab_file = get_vocab_file_from_task(task)

    base_config: Dict[str, Any] = {
        "behavior_version": 24,
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "recog_recomb": recomb_type,
        "recog_version": 10,
        "aux_loss_layers": [aux_ctc_layer] if aux_ctc_layer is not None else [],
    }
    if extra_config:
        base_config = dict_update_deep(base_config, extra_config)
    if ctc_soft_collapse_threshold is not None:
        base_config.update(
            {
                "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
                "ctc_soft_collapse_reduce_type": "max_renorm",
            }
        )

    if use_prior and labelwise_prior is None:
        prior = get_ctc_prior_probs_sharded(
            ctc_model,
            task.train_dataset.copy_train_as_static(),
            config={
                "behavior_version": 24,
                "batch_size": 200_000 * ctc_model.definition.batch_size_factor,
                "max_seqs": 2000,
                "aux_loss_layers": [aux_ctc_layer] if aux_ctc_layer is not None else [],
                **(extra_config or {}),
            },
            num_shards=num_shards,
            alias_prefix=f"{prefix}/prior",
        )
        tk.register_output(f"{prefix}/prior.txt", prior)
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label=ctc_ext_model_recog.output_blank_label, new_label_idx=_ctc_model_def_blank_idx
        ).out_vocab
        log_prior_wo_blank = PriorRemoveLabelRenormJob(
            prior_file=prior,
            prior_type="prob",
            vocab=vocab_w_blank_file,
            remove_label=ctc_ext_model_recog.output_blank_label,
            out_prior_type="log_prob",
        ).out_prior
        tk.register_output(f"{prefix}/log_prior_wo_blank.txt", log_prior_wo_blank)
        labelwise_prior = Prior(file=log_prior_wo_blank, type="log_prob", vocab=vocab_file)

    # Dev-phase fused search + LM score: the model carries the LM (for the fused scoring),
    # but the search itself detaches it (see _fused_search_lm_score_forward_step),
    # so lm_scale / prior do not matter here; they only apply to the first pass below.
    dev_model = get_ctc_with_lm_and_labelwise_prior(ctc_model=ctc_model, language_model=lm, lm_scale=0.0)
    dataset = task.dev_dataset

    # One fused (search + LM score) work item per shard, run across the node by the generic engine.
    work_items: Dict[str, Dict[str, Any]] = {}
    shard_keys: List[str] = []
    for s in range(num_shards):
        key = "shard_%03i" % s
        ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=s, seq_ordering="random")
        cfg = _fused_search_lm_score_config(
            dataset=ds, model_def=dev_model.definition, config={**base_config, "beam_size": n_best_list_size}
        )
        cfg = ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=dev_model.checkpoint.path,
            returnn_config=cfg,
            log_verbosity=5,
            device="gpu",
        )
        work_items[key] = {
            "returnn_config": cfg,
            "model_checkpoint": dev_model.checkpoint,
            "output_files": [_AM_OUT_FILENAME, _LM_OUT_FILENAME],
        }
        shard_keys.append(key)

    job = BatchedReturnnForwardJob(
        work_items,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    job.add_alias(f"{prefix}/dev-search-lmscore-batched")

    def merged(filename: str) -> tk.Path:
        """Single (num_shards=1) or merged (>1) output path for a given per-shard filename."""
        if num_shards == 1:
            return job.out_files[shard_keys[0]][filename]
        return MergeSearchOutputShardsJob(
            [job.out_files[k][filename] for k in shard_keys], output_gzip=filename.endswith(".gz")
        ).out_search_results

    am_scores = RecogOutput(output=merged(_AM_OUT_FILENAME))
    lm_scores = RecogOutput(output=merged(_LM_OUT_FILENAME))
    prior_scores = prior_score(am_scores, prior=labelwise_prior) if use_prior else None

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE/SPM to words
        am_scores = f(am_scores)
        lm_scores = f(lm_scores)
        prior_scores = f(prior_scores) if use_prior else None
        ref = f(ref)

    opt_scales_job = (
        ScaleTuningJob(
            scores={"am": am_scores.output, "prior": prior_scores.output, "lm": lm_scores.output},
            ref=ref.output,
            fixed_scales={"am": 1.0},
            negative_scales={"prior"},
            scale_relative_to={"prior": "lm"},
            evaluation="edit_distance",
        )
        if use_prior
        else ScaleTuningJob(
            scores={"am": am_scores.output, "lm": lm_scores.output},
            ref=ref.output,
            fixed_scales={"am": 1.0},
            evaluation="edit_distance",
        )
    )
    opt_scales_job.rqmt["engine"] = "short"
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales. Prior is still handled as negative in the 1stpass model below.
    prior_scale = opt_scales_job.out_real_scale_per_name["prior"] * (-1) if use_prior else None
    lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

    # First-pass CTC+LM+prior recog with the tuned scales, sharded over the node.
    first_pass_model = get_ctc_with_lm_and_labelwise_prior(
        ctc_model=ctc_model,
        prior=labelwise_prior.file if use_prior else None,
        prior_type=labelwise_prior.type if use_prior else "none",
        prior_scale=prior_scale,
        language_model=lm,
        lm_scale=lm_scale,
    )
    recog_config = {
        **base_config,
        "beam_size": first_pass_recog_beam_size,
        # Beam-scaled batch size, as in the non-batched pipeline,
        # but 4x its 20k base: that was fitted to 11 GB GPUs,
        # and on the 96 GB GH200s the eager frame loop is launch-latency-bound,
        # so a wider batch amortizes the per-frame kernel launches.
        "batch_size": int(
            80_000 * ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
        ),
    }
    score = _combined_recog_batched(
        prefix=prefix,
        task=task,
        model=first_pass_model,
        config=recog_config,
        num_shards=num_shards,
        recog_def=model_recog_with_recomb,
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", score.output)
    return score


def get_ctc_prior_probs_sharded(
    ctc_model,
    dataset,
    *,
    config: Optional[Dict[str, Any]] = None,
    num_shards: int,
    alias_prefix: Optional[str] = None,
) -> tk.Path:
    """
    CTC label prior (in prob space), like :func:`ctc_recog_ext.get_ctc_prior_probs`,
    but as ``num_shards`` work items of one :class:`BatchedReturnnForwardJob`
    plus an exact frame-count-weighted mean merge.
    """
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import tools_paths
    from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob, _ShardedDataset
    from i6_experiments.users.zeyer.collect_model_dataset_stats import (
        _collect_stats_returnn_forward_config_v2,
        _prior_mean_out_filename,
        _prior_info_out_filename,
    )
    from .aed_ctc import _aed_ctc_model_ctc_softmax_prior_returnn_forward

    work_items: Dict[str, Dict[str, Any]] = {}
    shard_keys: List[str] = []
    for s in range(num_shards):
        key = "shard_%03i" % s
        ds = _ShardedDataset(dataset, num_shards=num_shards, shard_index=s)
        cfg = _collect_stats_returnn_forward_config_v2(
            ds, ctc_model.definition, _aed_ctc_model_ctc_softmax_prior_returnn_forward, config=config
        )
        cfg = ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=ctc_model.checkpoint.path, returnn_config=cfg, log_verbosity=5, device="gpu"
        )
        work_items[key] = {
            "returnn_config": cfg,
            "model_checkpoint": ctc_model.checkpoint,
            "output_files": [_prior_mean_out_filename, _prior_info_out_filename],
        }
        shard_keys.append(key)
    job = BatchedReturnnForwardJob(
        work_items,
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    if alias_prefix:
        job.add_alias(f"{alias_prefix}-batched")
    merge = MergeShardedStatsMeanJob(
        means=[job.out_files[k][_prior_mean_out_filename] for k in shard_keys],
        infos=[job.out_files[k][_prior_info_out_filename] for k in shard_keys],
    )
    return merge.out_mean


class MergeShardedStatsMeanJob(Job):
    """
    Merge per-shard :class:`returnn.util.basic.Stats` mean dumps into one overall mean,
    weighting each shard by its total frame count (parsed from the info dump).
    """

    def __init__(self, *, means: List[tk.Path], infos: List[tk.Path]):
        super().__init__()
        assert means and len(means) == len(infos)
        self.means = means
        self.infos = infos
        self.out_mean = self.output_path("mean.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import re
        import numpy

        weighted_sum = None
        total_frames = 0
        for mean_f, info_f in zip(self.means, self.infos):
            mean = numpy.loadtxt(mean_f.get_path())
            info = open(info_f.get_path(), encoding="utf-8").read()
            # Stats.dump info line: "  <seqs> seqs, <frames> total frames, <avg> average ..."
            m = re.search(r"(\d+) seqs, (\d+) total frames", info)
            assert m, f"cannot parse frame count from {info_f.get_path()}: {info!r}"
            n = int(m.group(2))
            assert n > 0, f"shard {mean_f.get_path()}: zero frames?"
            weighted_sum = mean * n if weighted_sum is None else weighted_sum + mean * n
            total_frames += n
        numpy.savetxt(self.out_mean.get_path(), weighted_sum / total_frames)


def _fused_search_lm_score_config(*, dataset, model_def, config: Optional[Dict[str, Any]] = None):
    """
    Build the fused per-shard forward config: same as a v3 search config, but with our fused
    forward step (pure-CTC search -> collapse -> LM score) and our two-file forward callback.
    """
    from i6_experiments.users.zeyer.recog import search_config_v3
    from .ctc import model_recog_with_recomb

    config = dict(config or {})
    config["forward_step"] = _fused_search_lm_score_forward_step
    config["forward_callback"] = _fused_get_forward_callback
    return search_config_v3(
        dataset=dataset,
        model_def=model_def,
        recog_def=model_recog_with_recomb,
        config=config,
    )


def _fused_search_lm_score_forward_step(*, model, extern_data, **_kwargs_unused):
    """
    Run within RETURNN (eager forward). Pure-CTC search -> align->label collapse -> LM score.

    Marks three outputs: ``hyps`` (label seqs), ``am_scores``, ``lm_scores`` (each [batch, beam]).
    """
    import numpy as np
    import returnn.frontend as rf
    from returnn.tensor import Dim, batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.decoding.lm_rescoring import lm_rescore_def
    from .ctc import model_recog_with_recomb
    from .aed_ctc_batched import _ctc_collapse_wb

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()

    # Detach the LM during the search: without it, the recog def takes the pure-CTC path
    # (no per-step LM eval, no prior), giving the same N-best as the CTC-only search
    # of the non-batched pipeline. The LM then only scores that N-best below.
    lm = model.lm
    assert lm is not None
    model.lm = None
    try:
        hyps_wb, am_scores, out_spatial_dim, beam_dim = model_recog_with_recomb(
            model=model, data=data, data_spatial_dim=data_spatial_dim
        )
    finally:
        model.lm = lm
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
    label_spatial_dim = Dim(label_lens_t, name="lm_label_spatial")
    label_hyps = rf.convert_to_tensor(
        labels_arr, dims=[batch_dim, beam_dim, label_spatial_dim], dtype="int32", sparse_dim=model.target_dim, device=dev
    )

    # LM scores on the collapsed label hyps.
    lm_scores = lm_rescore_def(
        model=lm,
        targets=label_hyps,
        targets_beam_dim=beam_dim,
        targets_spatial_dim=label_spatial_dim,
    )

    run_ctx = rf.get_run_ctx()
    run_ctx.mark_as_output(label_hyps, "hyps", dims=[batch_dim, beam_dim, label_spatial_dim])
    run_ctx.mark_as_output(am_scores, "am_scores", dims=[batch_dim, beam_dim])
    run_ctx.mark_as_output(lm_scores, "lm_scores", dims=[batch_dim, beam_dim])


def _fused_get_forward_callback():
    """Forward callback writing two N-best score files: am.py.gz and lm.py.gz (same hyps)."""
    from typing import TextIO, Optional as _Optional
    from returnn.tensor import Tensor
    from returnn.forward_iface import ForwardCallbackIface

    class _FusedCallback(ForwardCallbackIface):
        def __init__(self):
            self.am_file: _Optional[TextIO] = None
            self.lm_file: _Optional[TextIO] = None

        def init(self, *, model):
            import gzip

            self.am_file = gzip.open(_AM_OUT_FILENAME, "wt", encoding="utf-8")
            self.am_file.write("{\n")
            self.lm_file = gzip.open(_LM_OUT_FILENAME, "wt", encoding="utf-8")
            self.lm_file.write("{\n")

        def process_seq(self, *, seq_tag: str, outputs):
            hyps: Tensor = outputs["hyps"]  # [beam, label_spatial]
            am_scores: Tensor = outputs["am_scores"]  # [beam]
            lm_scores: Tensor = outputs["lm_scores"]  # [beam]
            assert hyps.sparse_dim and hyps.sparse_dim.vocab  # from the model target_dim
            hyps_len = hyps.dims[1].dyn_size_ext  # [beam] or []
            num_beam = hyps.raw_tensor.shape[0]

            self.am_file.write(f"{seq_tag!r}: [\n")
            self.lm_file.write(f"{seq_tag!r}: [\n")
            for i in range(num_beam):
                n = hyps_len.raw_tensor[i] if hyps_len.raw_tensor.shape else hyps_len.raw_tensor
                hyp_ids = hyps.raw_tensor[i, :n]
                hyp_serialized = hyps.sparse_dim.vocab.get_seq_labels(hyp_ids)
                self.am_file.write(f"  ({float(am_scores.raw_tensor[i])!r}, {hyp_serialized!r}),\n")
                self.lm_file.write(f"  ({float(lm_scores.raw_tensor[i])!r}, {hyp_serialized!r}),\n")
            self.am_file.write("],\n")
            self.lm_file.write("],\n")

        def finish(self):
            self.am_file.write("}\n")
            self.am_file.close()
            self.lm_file.write("}\n")
            self.lm_file.close()

    return _FusedCallback()
