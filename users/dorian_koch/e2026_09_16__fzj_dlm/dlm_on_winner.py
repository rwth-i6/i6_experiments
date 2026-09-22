"""
DLM training data from Albert Zeyer's FZJ text-injection winner, with the paper-best DLM recipe.

Paper-best DLM (2512.13576): ``base-scalingLaws-enc24-dec8-n1280-nEp200`` (RZ ``ReturnnTrainingJob.LfAv45zfWLtF``,
DLM-sum 1.48/3.26/1.70/3.44 with the old CTC). Its data task is the ``low_task`` built in
``denoising_lm_2024/sis_recipe/dlm_scaling_laws.py::get_dlm_scaling_stats``; :func:`get_dlm_task_on_winner`
calls the same builder with the same arguments, except ``hyps_model`` = the winner (CTC from its aux layer 16).

JUPITER bills per node (4 GH200), so the hypothesis forwards must not run as single-GPU jobs.
The data builder creates them via ``forward_to_hdf``; :func:`_with_batched_gpu_forwards` builds the task twice:
pass 1 records every GPU ``forward_to_hdf`` call (CPU calls go through unchanged),
the recorded forwards are bundled into :class:`BatchedReturnnForwardJob` s (4 workers per node),
pass 2 rebuilds the task with each call answered by its bundled output path.
Pass 2 asserts that every call reproduces the recorded RETURNN config hash, in the same order,
and that no recorded call is left unreplayed (e.g. through a cached function).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from sisyphus import Job, tk

# Only len() of get_tts_oggzips() is used on this path (always NUM_CORPUS_FILES = 750); the zips themselves are
# not inputs of the LM-text pipeline, and their Paths hash via hash_overwrite. Without a base dir the
# lookup raises FileNotFoundError off RZ/i6 (denoising_lm_2024/sis_recipe/tts_data.py:41-66).
_TTS_BASE_DIR_PLACEHOLDER = "/nonexistent/denoising-lm/lm_tts_2024"


class PlaceholderForwardJob(Job):
    """
    Pass-1 stand-in for a GPU forward (see :func:`_with_batched_gpu_forwards`). Never runs: only jobs built in
    pass 1 depend on it, and nothing registered does. It exists because the data builder touches the creator of
    the returned HDF (``hdf.creator.rqmt.update(...)``, ``hdf.creator.add_alias(...)``).
    """

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.out_hdf = self.output_path("out.hdf")
        self.rqmt = {}

    def tasks(self):
        raise RuntimeError("PlaceholderForwardJob must never be scheduled")


def get_dlm_task_on_winner(
    *,
    hyps_model,
    extra_config: Optional[Dict[str, Any]],
    alias_prefix: str,
    items_per_job: int = 8,
):
    """
    :param hyps_model: the winner as recorded from its CTC+LM recog (``ModelWithCheckpoint``)
    :param extra_config: the winner's recog model config (``aux_loss_layers=[16]``, pseudo-encoder settings)
    :param alias_prefix:
    :param items_per_job: forwards bundled per 4-GPU node job
    :return: (task, batched forward jobs)
    """
    os.environ.setdefault("__DLM_TTS_BASE_DIR", _TTS_BASE_DIR_PLACEHOLDER)

    from denoising_lm_2024.sis_recipe.error_correction_model_gen_train_data import (
        get_error_correction_model_task_via_tts_txt,
        GetCtcHypsCfgV6,
    )
    from denoising_lm_2024.sis_recipe.tts_model import get_tts_opts_default_model
    from i6_experiments.users.zeyer.returnn.models.rf_mixup import MixupOpts

    def _build():
        # Verbatim the low_task arguments of dlm_scaling_laws.get_dlm_scaling_stats, except hyps_model and
        # get_hyps_extra_config (the RZ call passed {"behavior_version": 24}; the winner additionally needs its
        # recog model config, as in its own CTC recogs).
        return get_error_correction_model_task_via_tts_txt(
            train_epoch_split=20,
            vocab="spm10k",
            num_hyps=1,
            hyps_cfg=GetCtcHypsCfgV6(
                dropout_min=0.0,
                dropout_max=0.2,
                enable_specaugment=True,
                specaugment_opts={"steps": (0, 0, 0), "max_consecutive_feature_dims": 0},
                data_perturbation_opts={
                    "mixup": MixupOpts(max_num_mix=2, lambda_min=0.0, lambda_max=0.2, apply_prob=1),
                },
            ),
            hyps_model=hyps_model,
            hyps_tts_opts=get_tts_opts_default_model(
                {
                    "glow_tts_noise_scale_range": (0.3, 0.9),
                    "glow_tts_length_scale_range": (0.7, 1.1),
                },
                compatible_to_nick=True,
            ),
            additional_eval_sets=["trainlike-lm-devtrain"],
            train_repeat_asr_data=10,
            train_repeat_asr_data_via_num_hyps=True,
            register_output=False,
            resplit_subwords=False,
            dataset_use_deep_copy=True,
            get_hyps_extra_config={"behavior_version": 24, **(extra_config or {})},
            use_dependency_boundary=False,
        )

    return _with_batched_gpu_forwards(_build, alias_prefix=alias_prefix, items_per_job=items_per_job)


def _with_batched_gpu_forwards(
    build: Callable[[], Any], *, alias_prefix: str, items_per_job: int
) -> Tuple[Any, List[Any]]:
    import unittest.mock
    from sisyphus.hash import short_hash
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_experiments.users.zeyer import forward_to_hdf as fth
    from i6_experiments.users.zeyer.forward_batched import BatchedReturnnForwardJob
    from i6_experiments.users.zeyer.returnn.config import pop_from_config_post_config

    orig_forward_to_hdf = fth.forward_to_hdf

    def _is_gpu(kw: Dict[str, Any]) -> bool:
        return (kw.get("forward_device") or ("gpu" if kw.get("model") else "cpu")) == "gpu"

    def _work_item(kw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Same config construction as forward_to_hdf, as a BatchedReturnnForwardJob work item."""
        assert kw.get("_config_v2", True), "only the v2 config path is mirrored"
        assert not (kw.get("forward_def") and kw.get("forward_step"))
        config, post_config, env_updates = pop_from_config_post_config(
            kw.get("config"), kw.get("forward_post_config"), key="__env_updates"
        )
        config, post_config, _rqmt = pop_from_config_post_config(
            config, post_config, key="__rqmt_updates", prev=kw.get("forward_rqmt")
        )
        model = kw.get("model")
        cfg = fth._returnn_forward_config_v2(
            dataset=kw["dataset"],
            model_def=model.definition if model else None,
            forward_def=kw.get("forward_def"),
            forward_step=kw.get("forward_step"),
            config=config,
            post_config=post_config,
        )
        cfg = ReturnnForwardJobV2.create_returnn_config(
            model_checkpoint=model.checkpoint.path if model else None,
            returnn_config=cfg,
            log_verbosity=5,
            device="gpu",
        )
        item = {
            "returnn_config": cfg,
            "output_files": [fth._hdf_out_filename],
            "model_checkpoint": model.checkpoint if model else None,
        }
        return item, dict(env_updates or {})

    # Pass 1: record.
    recorded: List[Tuple[Dict[str, Any], Dict[str, str]]] = []

    def _record(**kw):
        if not _is_gpu(kw):
            return orig_forward_to_hdf(**kw)
        recorded.append(_work_item(kw))
        return PlaceholderForwardJob(len(recorded)).out_hdf

    # The builder registers outputs itself (even with register_output=False), which would put pass-1 placeholder
    # jobs and every bundle into the graph. Suppress it in both passes; the caller registers what should run.
    no_register = unittest.mock.patch.object(tk, "register_output", lambda *_args, **_kwargs: None)
    with unittest.mock.patch.object(fth, "forward_to_hdf", _record), no_register:
        build()
    assert recorded, "the data builder made no GPU forward_to_hdf call"

    # Bundle into full-node jobs. Keys are local to the bundle, so a bundle's hash depends only on its items.
    jobs: List[Any] = []
    paths: List[tk.Path] = []
    for b, start in enumerate(range(0, len(recorded), items_per_job)):
        chunk = recorded[start : start + items_per_job]
        work_items = {f"item_{j:02d}": item for j, (item, _env) in enumerate(chunk)}
        job = BatchedReturnnForwardJob(work_items)
        env: Dict[str, str] = {}
        for _item, item_env in chunk:
            env.update(item_env)
        for k, v in sorted(env.items()):
            job.set_env(k, v)  # BatchedReturnnForwardJob does not apply __env_updates itself
        job.add_alias(f"{alias_prefix}/hyps-batched-{b:02d}")
        jobs.append(job)
        paths.extend(job.out_files[key][fth._hdf_out_filename] for key in work_items)

    # Pass 2: replay.
    # The builder calls hdf.creator.rqmt.update({"time": 48, "mem": 16}) per forward (ctc.py:1009); on a bundle that
    # would shrink a full-node job to 16 GB host RAM. Restore each bundle's own rqmt afterwards (same dict object).
    saved_rqmt = [dict(job.rqmt) for job in jobs]
    replayed = [0]

    def _replay(**kw):
        if not _is_gpu(kw):
            return orig_forward_to_hdf(**kw)
        i = replayed[0]
        replayed[0] += 1
        assert i < len(recorded), f"pass 2 made more GPU forwards than pass 1 ({i + 1} > {len(recorded)})"
        item, _env = _work_item(kw)
        assert short_hash(item["returnn_config"]) == short_hash(recorded[i][0]["returnn_config"]), (
            f"GPU forward #{i} differs between the two passes"
        )
        return paths[i]

    with unittest.mock.patch.object(fth, "forward_to_hdf", _replay), no_register:
        task = build()
    assert replayed[0] == len(recorded), (
        f"pass 2 replayed {replayed[0]} of {len(recorded)} GPU forwards (cached builder function?)"
    )
    for job, rqmt in zip(jobs, saved_rqmt):
        job.rqmt.clear()
        job.rqmt.update(rqmt)
    return task, jobs


def describe_bundles(jobs: List[Any]) -> None:
    """Print what each bundled 4-GPU job forwards (for inspection before submitting)."""
    total = 0
    for b, job in enumerate(jobs):
        print(f"bundle {b:02d}: {len(job.work_items)} items  {job._sis_id()}")
        for key, item in job.work_items.items():
            total += 1
            cfg = item["returnn_config"]
            c = getattr(cfg, "config", {}) or {}
            fd = c.get("forward_data") or {}
            desc = {k: fd.get(k) for k in ("class", "partition_epoch", "seq_ordering") if isinstance(fd, dict) and k in fd}
            files = []
            if isinstance(fd, dict):
                for k in ("corpus_file", "path", "files", "seq_list_file"):
                    if k in fd:
                        v = fd[k]
                        v = v if isinstance(v, (list, tuple)) else [v]
                        files += [str(getattr(x, "path", x)).split("/")[-3:] and "/".join(str(getattr(x, "path", x)).split("/")[-3:]) for x in v][:3]
            print(f"   {key}: {desc} {files} random_seed={c.get('random_seed')}")
    print(f"total GPU forwards: {total}")


DLM_NAME = "base-scalingLaws-enc24-dec8-n1280-nEp200"


def train_paper_best_dlm_4gpu(task, *, name_suffix: str = "-winnerHyps-4gpu", model_dim: int = 1280):
    """
    The paper-best DLM (``dlm_scaling_laws.get_dlm_scaling_stats``, entry (24, 8, 1280), nEp 200) on ``task``,
    as 4-GPU DDP on one JUPITER node with the same optimization as the single-GPU RZ run:

    - effective batch 20k tokens: 4 ranks x ``batch_size`` 5k (``max_seqs`` 2000 -> 500 per rank),
    - same steps / data passes: RETURNN DDP ranks each iterate the full sub-epoch partition (per-rank seed
      offset, no sharding), so 4 ranks cover 4 sub-epochs of data per sub-epoch -> nEp 200 / 4 = 50
      (Albert's FZJ convention, see exp2026_05_28_tts_encoder_fzj.py), ~919k steps either way,
    - LR schedule ``_get_cfg_lrlin_oclr_by_bs_nep_v4`` is by epoch fraction, so it scales with nEp.
    No per-epoch recog here (train_exp's would be single-GPU jobs); evaluate with the batched DLM-sum instead.

    :param model_dim: 1280 = the paper-best entry (name and hash unchanged). Any other value changes ONLY the
        width, e.g. 1024 (~466M params) for a DLM near the size of the n32-d1024 LM (~422M) it is compared to.
    """
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        _get_cfg_lrlin_oclr_by_bs_nep_v4,
    )
    import returnn.frontend as rf
    from returnn.frontend.encoder.transformer import TransformerEncoder
    from returnn.frontend.decoder.transformer import TransformerDecoder
    from denoising_lm_2024.sis_recipe.error_correction_model_claix2023_dorian import (
        train_exp,
        aed_model_def,
        common_trafo_enc_kwargs,
        common_trafo_dec_kwargs,
        train_base_cfg,
    )

    num_enc, num_dec, n_epochs, num_gpus = 24, 8, 200, 4
    additional_opts = {"model_dim": model_dim}
    name = DLM_NAME if model_dim == 1280 else DLM_NAME.replace("-n1280-", f"-n{model_dim}-")
    return train_exp(
        name + name_suffix,
        task,
        model_def=ModelDefWithCfg(
            aed_model_def,
            {
                "_encoder_model_dict": rf.build_dict(
                    TransformerEncoder, num_layers=num_enc, **dict_update_deep(common_trafo_enc_kwargs, additional_opts)
                ),
                "_decoder_model_dict": rf.build_dict(
                    TransformerDecoder, num_layers=num_dec, **dict_update_deep(common_trafo_dec_kwargs, additional_opts)
                ),
                "input_add_eos": True,
            },
        ),
        config=dict_update_deep(
            train_base_cfg,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_epochs // num_gpus),
                "batch_size": 20_000 // num_gpus,
                "max_seqs": 2_000 // num_gpus,
                "min_seq_length": 1,
                "__multi_proc_dataset": False,
                "input_swapout_range": (0.1, 0.1),
                # 4-GPU DDP on one node (without torch_distributed, num_processes>1 falls back to horovod)
                "torch_distributed": {},
                "__num_processes": num_gpus,
                "__gpu_mem": 96,
                "__mem_rqmt": 100,  # per rank; sis multiplies by num_processes
                "__cpu_rqmt": 72,  # per rank; 4 x 72 = the full node
            },
        ),
        post_config={"log_grad_norm": True},
        with_recog=False,
    )
