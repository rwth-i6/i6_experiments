"""
DLM on top of Albert Zeyer's FZJ text-injection ASR winner.

Step 1 (this file today): rebuild the winner's graph with Albert's own builder, so every job gets
the hash it has in his setup, and import his finished jobs instead of recomputing them.
Nothing here is meant to be run by the manager yet.

The winner and all its recogs come from ONE builder call, copied verbatim from the ``_sa == 50``
iteration of the specaug loop in ``exp2026_05_28_tts_encoder_fzj.py::py`` -- including
``with_ctc_lm_recog=True``, which wires CTC+LM, AED+CTC+LM label-sync and both DLM-sum recogs
(with the RZ DLM ``evKXenj1Z2j3`` from his ``import/dlm/``).

Import, from the setup root (loads the config, links finished jobs, submits nothing)::

    wrap.sh python ./sis c --script recipe/i6_experiments/users/dorian_koch/e2026_09_16__fzj_dlm/fzj_dlm.py \\
        -c 'from i6_experiments.users.dorian_koch.e2026_09_16__fzj_dlm.fzj_dlm import import_albert_jobs, report' \\
        -c 'import_albert_jobs("dryrun")' -c 'report()'

(``"symlink"`` instead of ``"dryrun"`` to actually link.)
Project notes: ``projects/2026-09-16-fzj-dlm.md`` in the local remote-setup repo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sisyphus import tk

import returnn.frontend as rf

__all__ = ["py", "import_albert_jobs", "report", "report_outputs"]
__setup_root_prefix__ = "exp2026_09_16_fzj_dlm"

# Albert's readable work trees. Many job dirs in the first are symlinks into
# /e/home/jusers/zeyer1/... (unreadable for us); the second holds the readable copies of those.
ALBERT_WORK_DIRS: List[str] = [
    "/e/project1/spell/zeyer1/setups/2026-05-28-tts-encoder/work",
    "/e/project1/spell/zeyer1/setups/2026-05-26-base/work",
]

WINNER_NAME = "pseudo-enc-logmel-mfatable-realdur2-lerp-dur07-packed-single-gumbel-muon-nep38-specaug50-stepcomp"
WINNER_TRAIN_JOB = "i6_core/returnn/training/ReturnnTrainingJob.8iFbool3x3TU"

# DLM data from the winner (paper-best DLM recipe, see dlm_on_winner.py):
#   "off"   -- not built
#   "smoke" -- built; only the first 4-GPU hypothesis bundle is registered (verify GPU use + outputs first)
#   "hyps"  -- all hypothesis bundles registered
#   "train" -- all bundles + the paper-best DLM trained on them (4-GPU DDP), user decision 2026-09-16 22:20
DLM_DATA_STAGE = "train"
# Continue-train the winner with GlowTTS TTS audio added to its paired-audio branch (winner_plus_tts.py).
WINNER_PLUS_TTS = True
_dlm_hyp_jobs: List[Any] = []
_dlm_task_ref: List[Any] = []  # the DLM data task, for console inspection

# DLMs relayed from our RZ setup (rsync RZ -> FZJ, sha256-verified), by model_dim.
OUR_DLM_IMPORT_DIR = "/e/project1/spell/koch13/setups/2026-09-16-fzj-dlm/import/dlm"
DLM_N1280 = OUR_DLM_IMPORT_DIR + (
    "/base-scalingLaws-enc24-dec8-n1280-nEp200.ReturnnTrainingJob.LfAv45zfWLtF/output/models/epoch.200.pt"
)


def _get_dlm(checkpoint: str, *, model_dim: int):
    """
    A DLM as an explicit-checkpoint model, same structure as Albert's ``_get_imported_dlm``
    (24-layer encoder / 8-layer decoder, RMSNorm, gated FF, rotary attention, input EOS),
    which read its dicts off the RZ training job's returnn.config. Only ``model_dim`` varies
    between our DLMs; checked against each job's returnn.config.
    """
    from i6_core.returnn.training import PtCheckpoint
    from i6_experiments.users.zeyer.model_interfaces.model import ModelDefWithCfg
    from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoint
    from denoising_lm_2024.error_correction_model import aed_model_def

    trafo_kwargs = {
        "model_dim": model_dim,
        "pos_enc": None,
        "norm": {"class": "rf.RMSNorm"},
        "ff": {"class": "returnn.frontend.decoder.transformer.FeedForwardGated"},
        "dropout": 0.0,
        "att_dropout": 0.0,
    }
    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(
            aed_model_def,
            {
                "_encoder_model_dict": {
                    "class": "returnn.frontend.encoder.transformer.TransformerEncoder",
                    "num_layers": 24,
                    "layer_opts": {"self_att": {"class": "rf.RotaryPosSelfAttention", "with_bias": False}},
                    **trafo_kwargs,
                },
                "_decoder_model_dict": {
                    "class": "returnn.frontend.decoder.transformer.TransformerDecoder",
                    "num_layers": 8,
                    "layer_opts": {"self_att": {"class": "rf.RotaryPosCausalSelfAttention", "with_bias": False}},
                    **trafo_kwargs,
                },
                "input_add_eos": True,
            },
        ),
        checkpoint=PtCheckpoint(tk.Path(checkpoint)),
    )


def py():
    from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

    prefix = get_setup_prefix_for_module(__name__)

    # Record the task / model / config the builder hands its CTC+LM recog (imported at call time from the
    # module), to reuse them verbatim for a plain CTC recog below instead of re-deriving them.
    import unittest.mock
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext import ctc_lm_batched as _ctc_lm

    ctc_lm_kwargs: Dict[str, Any] = {}
    _orig_ctc_lm = _ctc_lm.ctc_recog_recomb_labelwise_prior_auto_scale_batched

    def _record_ctc_lm(**kwargs):
        ctc_lm_kwargs.update(kwargs)
        return _orig_ctc_lm(**kwargs)

    with unittest.mock.patch.object(_ctc_lm, "ctc_recog_recomb_labelwise_prior_auto_scale_batched", _record_ctc_lm):
        winner_model = _train_winner(prefix)
    assert ctc_lm_kwargs, "builder did not call ctc_recog_recomb_labelwise_prior_auto_scale_batched"

    # Plain CTC (no LM, no prior) for the result table, 2026-09-16.
    _ctc_only_recog_batched(
        prefix=f"{prefix}/aed/{WINNER_NAME}/ctc-only-batched",
        task=ctc_lm_kwargs["task"],
        ctc_model=ctc_lm_kwargs["ctc_model"],
        aux_ctc_layer=ctc_lm_kwargs["aux_ctc_layer"],
        num_shards=ctc_lm_kwargs["num_shards"],
        extra_config=ctc_lm_kwargs.get("extra_config"),
    )

    if DLM_DATA_STAGE != "off":
        from .dlm_on_winner import get_dlm_task_on_winner

        _dlm_task, jobs = get_dlm_task_on_winner(
            hyps_model=ctc_lm_kwargs["ctc_model"],
            extra_config=ctc_lm_kwargs.get("extra_config"),
            alias_prefix=f"{prefix}/dlm-data",
        )
        _dlm_hyp_jobs[:] = jobs
        _dlm_task_ref[:] = [_dlm_task]
        # smoke: bundle 01 = 2 LS-960 passes (ogg audio) + 6 LM-text shards (GlowTTS path), i.e. both code paths
        register = {1: jobs[1]} if DLM_DATA_STAGE == "smoke" else dict(enumerate(jobs))
        for b, job in register.items():
            for key, outs in job.out_files.items():
                for fn, path in outs.items():
                    tk.register_output(f"{prefix}/dlm-data/hyps-batched-{b:02d}/{key}/{fn}", path)
        if DLM_DATA_STAGE == "train":
            from .dlm_on_winner import train_paper_best_dlm_4gpu

            train_paper_best_dlm_4gpu(_dlm_task)

    # Continue training the winner with TTS audio added (user request, 2026-09-17). Independent of the DLM
    # line above: it only adds jobs, and it touches none of tts_data's module state, so the hypothesis
    # bundles now running keep their hashes.
    if WINNER_PLUS_TTS:
        from .winner_plus_tts import train_winner_plus_tts

        train_winner_plus_tts(prefix=f"{prefix}/winner-plus-tts", winner_model=winner_model)

    # End-to-end check (2026-09-16): the same winner recogs with the 729M DLM (LfAv45zfWLtF).
    # The builder looks up _get_imported_dlm at call time, so swapping it makes the DLM-sum recogs
    # the only new jobs; everything else resolves to the identical (imported) jobs.
    from i6_experiments.users.zeyer.experiments import exp2026_05_28_tts_encoder_fzj as _fzj

    with unittest.mock.patch.object(_fzj, "_get_imported_dlm", lambda: _get_dlm(DLM_N1280, model_dim=1280)):
        _train_winner(prefix + "/dlm-n1280")


def _ctc_only_recog_batched(
    *,
    prefix: str,
    task,
    ctc_model,
    aux_ctc_layer: Optional[int],
    num_shards: int,
    extra_config: Optional[Dict[str, Any]] = None,
    beam_size: int = 64,
    ctc_soft_collapse_threshold: Optional[float] = 0.8,
):
    """
    Plain CTC first pass: time-synchronous recombination beam search, no LM, no prior.

    The first pass of ``ctc_lm_batched.ctc_recog_recomb_labelwise_prior_auto_scale_batched`` with the LM
    left out: same search def (``recog_ext.ctc.model_recog_with_recomb``, which skips LM and prior when
    ``model.lm`` is None), same base config, beam, soft collapse and beam-scaled batch size.
    """
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc_batched import (
        _combined_recog_batched,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc import model_recog_with_recomb

    config: Dict[str, Any] = {
        "behavior_version": 24,
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "recog_recomb": "max",
        "recog_version": 10,
        "aux_loss_layers": [aux_ctc_layer] if aux_ctc_layer is not None else [],
    }
    if extra_config:
        config = dict_update_deep(config, extra_config)
    if ctc_soft_collapse_threshold is not None:
        config.update(
            {"ctc_soft_collapse_threshold": ctc_soft_collapse_threshold, "ctc_soft_collapse_reduce_type": "max_renorm"}
        )
    config["beam_size"] = beam_size
    config["batch_size"] = int(40_000 * ctc_model.definition.batch_size_factor * min(32 / beam_size, 1))
    score = _combined_recog_batched(
        prefix=prefix,
        task=task,
        model=ctc_model,
        config=config,
        num_shards=num_shards,
        recog_def=model_recog_with_recomb,
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", score.output)
    return score


def _train_winner(prefix: str):
    """The ``_sa == 50`` call of the specaug loop in ``exp2026_05_28_tts_encoder_fzj.py::py``, verbatim.

    Returns the ``ModelWithCheckpoints`` so a follow-up experiment can init from a real checkpoint of it
    (see ``winner_plus_tts``) rather than naming a path.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import (
        get_mfa_phone_mean_logmel_table,
        get_mfa_phone_duration_table,
    )
    from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder

    _sa = 50
    return _train_tts_encoder(
        WINNER_NAME,
        prefix=prefix,
        with_ctc_lm_recog=True,
        text_train_epoch_split=75,
        batch_size_audio_frames=70_000,
        batch_size_phon=6_000,
        max_phon_len=300,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
        pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
        pseudo_enc_duration_sigma=0.45,
        pseudo_enc_duration_scale=0.7,
        pseudo_enc_max_len_factor=10,
        train_seq_ordering="random",
        pseudo_enc_lerp=True,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        glow_tts_add_silence_between_words=0.15,
        base_lr=1.0,
        peak_lr=5e-3,
        nep=38,
        behavior_version=29,  # packed tensors need >= 29
        pseudo_enc_frontend_concat=True,
        extra_config_updates={
            "optimizer.class": rf.build_dict(Muon)["class"],
            "packed_tensors": True,
            "torch_distributed": {"reduce_type": "grad_explicit"},
            "batch_size": None,
            "packed_batch_size": {"data": 11_200_000, "classes": 5_000, "phonemes": 6_000},
            "batching": "random",
            "torch_cuda_graph": {
                "batch_size_bound": 500,
                "dim_capacity": {"data": 312_000, "classes": 80, "phonemes": 300},
                "warmup_steps": 0,
                "compile": True,
            },
            "optimizer.weight_decay": 0.027,  # 0.01 / 0.370
            "specaugment_num_spatial_mask_factor": _sa,
            "specaugment_steps": (1850, 5550, 9250),  # (5000, 15000, 25000) * 0.370
        },
        extra_config_deletes=["optimizer.epsilon"],
    )


def import_albert_jobs(mode: str = "dryrun"):
    """Link Albert's finished jobs into our work/ (mode: dryrun | symlink | copy | hardlink)."""
    tk.import_work_directory(ALBERT_WORK_DIRS, mode=mode)


def report(max_listed: int = 60):
    """Print, for every job in the loaded graph: whether it exists in our work/ and is finished."""
    import os

    jobs = tk.sis_graph.jobs()
    counts: Dict[str, int] = {}
    missing = []
    for job in jobs:
        setup, finished = job._sis_setup(), job._sis_setup() and job._sis_finished()
        linked = os.path.islink(job._sis_path())
        key = "finished" + (" (symlink)" if linked else "") if finished else ("setup, unfinished" if setup else "absent")
        counts[key] = counts.get(key, 0) + 1
        if not finished:
            missing.append(job)
    print(f"graph: {len(jobs)} jobs")
    for key, n in sorted(counts.items()):
        print(f"  {n:6d}  {key}")

    winner = [j for j in jobs if j._sis_id() == WINNER_TRAIN_JOB]
    print(f"winner training job {WINNER_TRAIN_JOB}: " + ("IN GRAPH" if winner else "NOT IN GRAPH (hash mismatch?)"))
    if winner:
        w = winner[0]
        models = os.path.join(w._sis_path(), "output", "models")
        present = sorted(os.listdir(models)) if os.path.isdir(models) else []
        print(f"  finished={w._sis_setup() and w._sis_finished()}  checkpoints={present}")

    print(f"not finished here ({len(missing)}), first {max_listed}:")
    for job in sorted(missing, key=lambda j: j._sis_id())[:max_listed]:
        aliases = sorted(job.get_aliases() or [])
        print(f"  {job._sis_id()}  {aliases[0] if aliases else ''}")


def report_outputs(name_substr: str = "", max_bytes: int = 2000):
    """Print every registered output: available or not, and the content of small ones (WER summaries).
    Works from `sis c` without a manager, i.e. without creating output/ links."""
    import os
    from sisyphus.graph import OutputPath

    for target in sorted(tk.sis_graph.targets, key=lambda t: t.name):
        if name_substr not in target.name or not isinstance(target, OutputPath):
            continue
        path = target._sis_path.get_path()
        ok = target._sis_path.available()
        line = f"{'OK ' if ok else 'MISSING'}  {target.name}"
        if ok and os.path.isfile(path) and os.path.getsize(path) <= max_bytes:
            with open(path) as f:
                line += "  " + " ".join(f.read().split())
        print(line)
