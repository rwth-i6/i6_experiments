"""
Shared builders for the phoneme Conformer acoustic-model baselines (pHMM / CTC).

The three phon AM experiments -- EOW-phoneme pHMM (``phmm_phon.baseline``), EOW-phoneme CTC
(``ctc_phon.baseline``) and non-EOW monophone pHMM (``phmm_phon.baseline_noeow``) -- are identical
except for the differences their names imply:

* the phoneme inventory / lexicon (EOW-augmented vs. plain monophone; ``[SILENCE]`` vs. ``[BLANK]``
  at index 0),
* the training-FSA topology (``hmm`` vs. ``ctc``),
* the recognition config (HMM vs. CTC label collapsing).

Everything else -- the Conformer model, feature extraction, SpecAugment, the optimizer / OCLR-style
/4 learning-rate schedule, the 4-GPU ``torch_distributed`` parameter-averaging, bf16 AMP, the aux
CTC losses, label smoothing, the data settings, and the lexicon-constrained recognition driver --
lives here exactly once and is shared verbatim. This module is the single source of truth for those
shared hyperparameters; the per-experiment entry points only assemble the few name-implied pieces
and call :func:`train_and_lexicon_search`.
"""
import copy
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sisyphus import tk

from .data.common import DatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, RETURNN_ROOT, LIBRASR_WHEEL
from .pipeline import training, prepare_asr_model, search, compute_per
from .results import add_result
from .rasr import CreateLibrasrVenvJob

# --- Shared constants ---------------------------------------------------------------------------
# All three phon AMs are 4-GPU, single-node, data-parallel (parameter averaging). gpu_mem is a
# Sisyphus job *requirement* (NOT hashed) -- the batch_size is the 24GB-tuned value and is hashed,
# so the GPU memory tier can be switched freely (just change GPU_MEM) without rehashing the training.
NUM_GPUS = 4
GPU_MEM = 48

NETWORK_MODULE = "phmm.phmm_zhou"
# AM model name shared by all three experiments; the per-experiment suffix (e.g. "_ctc") is appended
# by the caller. "lr3e-04" == f"{3e-4:.0e}" (the single peak LR swept below).
BASE_MODEL_NAME = NETWORK_MODULE + ".512dim_sub4_50eps_sp_lp_fullspec_gradnorm_radam_lr3e-04"

# Checkpoints kept + evaluated (the full /4 = 125-subepoch schedule).
DEFAULT_CKPT_LIST = [10, 20, 40, 60, 80, 100, 110, 125]

# Seconds of audio per AM **output** frame: the log-mel hop is 10 ms and the VGG frontend subsamples
# the time axis by 4 (two stride-2 poolings, see build_conformer_model_config), so each emitted frame
# advances 40 ms. Used to convert the LibRASR search traceback's frame-index segment boundaries into
# real time for the phoneme/silence duration analysis (see phon_lexfree.duration eval).
AM_FRAME_SHIFT_SECONDS = 0.01 * 4

# SpecAugment turns on at subepoch 11 (NOT the /4-quartered 3). The pHMM alignment breakthrough is
# gated by the number of CLEAN (unmasked) optimizer steps before masking: 1-GPU and 4-GPU both do
# ~1979 steps/subepoch, so specaug@3 masks after only ~6k clean steps (loss still ~1.53) and the
# alignment never crystallizes, whereas specaug@11 gives ~22k clean steps (down to ~1.34) and it
# breaks through. The /4 schedule wrongly quartered specauc_start_epoch 11->3 to hold DATA constant,
# but the breakthrough depends on STEPS. (Proven 2026-05-31 by the config_06-12 elimination chain.)
SPECAUG_START_EPOCH = 11


def build_phon_train_settings() -> DatasetSettings:
    """Data settings shared by training and the dev/test recognition sets."""
    return DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )


def make_librasr_returnn() -> Dict[str, Any]:
    """The LibRASR venv + real-RETURNN root used for all phon training/recognition."""
    phmm_returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        # "ninja" for i6_native_ops JIT; "dm-tree"/"h5py" are real-RETURNN deps that MiniReturnn
        # did not require (dm-tree provides `tree`, imported by returnn.frontend at startup).
        extra_pip_packages=["ninja", "dm-tree", "h5py"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin
    return {"returnn_exe": phmm_returnn_exe, "returnn_root": RETURNN_ROOT}


def build_dev_test_dataset_tuples(settings: DatasetSettings):
    """``(dev_dataset_tuples, test_dataset_tuples)`` -- the standard LibriSpeech dev/test sets."""
    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(dataset_key=testset, settings=settings)
    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(dataset_key=testset, settings=settings)
    return dev_dataset_tuples, test_dataset_tuples


def build_conformer_model_config(label_target_size, specauc_start_epoch: int = SPECAUG_START_EPOCH):
    """The 12x512 Conformer model config; only ``label_target_size`` differs across experiments."""
    from .pytorch_networks.phmm.phmm_zhou_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        ConformerPosEmbConfig,
    )

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # classic style
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )
    return ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        pos_emb_config=posemb_config,
        specaug_config=specaug_config,
        label_target_size=label_target_size,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        mhsa_with_bias=True,
        conv_kernel_size=31,
        final_dropout=0.05,
        specauc_start_epoch=specauc_start_epoch,
        dropout_broadcast_axes=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=[5, 11],
        aux_ctc_loss_scales=[0.2, 0.8],
    )


def build_phon_train_config(ckpt_list: List[int], peak_lr: float = 3e-4, init_lr: float = 1e-5) -> Dict[str, Any]:
    """
    The RADAM / OCLR-style /4 schedule / 4-GPU parameter-averaging / bf16 training config shared by
    all three phon AMs.
    """
    return {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        # schedule = single-GPU 240/240/20 (=500) divided by NUM_GPUS=4 -> 60/60/5 (=125 subepochs).
        # peak_lr is NOT scaled: torch_distributed param-averaging is not a true large-batch step.
        "learning_rates": list(np.linspace(init_lr, peak_lr, 60))
        + list(np.linspace(peak_lr, init_lr, 60))
        + list(np.linspace(init_lr, 1e-7, 5)),
        # Single-node multi-GPU via parameter averaging every 100 steps (the chosen mode -- the
        # config_06-12 saga proved param-averaging is NOT what broke the 4-GPU run; specaug timing was).
        "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},
        #############
        # extern_data is required by real RETURNN. "labels" maps to the OggZip "orth" stream, i.e. the
        # raw UTF-8 bytes of the orthography (sparse dim 256, uint8); the train step decodes these bytes
        # back to text for the RASR FSA builder.
        "behavior_version": 21,
        "extern_data": {
            "raw_audio": {"dim": 1},
            "labels": {"dim": 256, "sparse": True, "dtype": "uint8"},
        },
        "batch_size": 400 * 16000,
        "max_seq_length": {"raw_audio": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 10.0,
        # bf16 has fp32's exponent range, so a GradScaler is pointless here. real-RETURNN's default
        # (grad_scaler={}) only (a) inflates the logged grad_norm -- it's logged pre-unscale, so it
        # tracks the auto-ramping loss-scale (2^16 -> 2^50+), not the true gradient -- and (b) silently
        # skips any inf/nan-grad step. Disable it. Because the FSA can emit inf losses (infeasible
        # T/4 < #phones) whose torch.where mask leaks a nan gradient, replace the scaler's silent skip
        # with RETURNN's explicit, *logged* nan-grad guard so we don't trade a cosmetic issue for a crash.
        "torch_amp": {"dtype": "bfloat16", "grad_scaler": None},
        "num_allowed_consec_invalid_gradient_steps": 10,
        "torch_dataloader_opts": {"num_workers": 2},
        "log_grad_norm": True,
        "cleanup_old_models": {"keep": ckpt_list},
    }


def train_and_lexicon_search(
    *,
    prefix_name: str,
    train_subdir: str,
    model_name_suffix: str,
    train_data,
    vocab_size_without_blank,
    fsa_exporter_config,
    returnn: Dict[str, Any],
    recog_rasr_config=None,
    recog_lexicon=None,
    dev_dataset_tuples=None,
    test_dataset_tuples=None,
    decoder_silence_label: Optional[str] = None,
    per_lexicon=None,
    specauc_start_epoch: int = SPECAUG_START_EPOCH,
    ckpt_list: Optional[List[int]] = None,
    run_lexicon_search: bool = True,
) -> Dict[str, Any]:
    """
    Train the shared Conformer AM and (optionally) run the per-epoch lexicon-constrained recognition
    with the 4-gram word LM at a fixed lm.scale=1.0, feeding WERs (and optional auxiliary PER) into
    the combined ``summary.report``.

    Name-implied per-experiment knobs:

    :param train_subdir: training-artifact subdir under ``prefix_name`` (e.g. ``"eow_phon"`` / ``"phon"``).
    :param model_name_suffix: appended to :data:`BASE_MODEL_NAME` (e.g. ``""`` / ``"_ctc"``).
    :param fsa_exporter_config: the topology-specific (hmm/ctc) FSA exporter config for the train step.
    :param recog_rasr_config / recog_lexicon: the topology-specific recognition config + recog lexicon
        (required only when ``run_lexicon_search=True``).
    :param decoder_silence_label: ``[BLANK]`` for CTC (index-0 blank lemma), ``None`` for the HMM
        ``[SILENCE]`` default.
    :param per_lexicon: if given, also compute an auxiliary PER by phonemizing the WORD hypothesis with
        this (training) lexicon -- a like-for-like phoneme comparison vs. the lexicon-free path.
    :param run_lexicon_search: if False, only train + build ``asr_models_by_epoch`` (the caller then
        runs its own recognition, e.g. a prior + LM-scale sweep).
    :return: artifacts dict consumed by the lexicon-free recognition entry points.
    """
    from .pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig

    if ckpt_list is None:
        ckpt_list = list(DEFAULT_CKPT_LIST)

    model_config = build_conformer_model_config(vocab_size_without_blank, specauc_start_epoch)
    train_config = build_phon_train_config(ckpt_list)

    train_args = {
        "config": train_config,
        "network_module": NETWORK_MODULE,
        "net_args": {"model_config_dict": asdict(model_config)},
        "train_step_args": {
            "fsa_exporter_config_path": fsa_exporter_config,
            "label_smoothing_scale": 0.1,
        },
        "include_native_ops": True,
        "use_speed_perturbation": True,
        "debug": False,
    }

    model_name = BASE_MODEL_NAME + model_name_suffix
    # Training artifacts live under <train_subdir>/<model>; recognition results live under
    # lexicon-search/<model> and feed the combined summary.report (see results.add_result).
    training_name = prefix_name + "/" + train_subdir + "/" + model_name
    lexicon_search_root = prefix_name + "/lexicon-search/" + model_name

    train_job = training(
        training_name,
        train_data,
        train_args,
        num_epochs=125,  # single-GPU 500 / NUM_GPUS=4
        num_processes=NUM_GPUS,
        distributed_launch_cmd="torchrun",
        **returnn,
    )
    # gpu_mem is a (non-hashed) requirement -> switch GPU tier here without rehashing.
    # i6_core already scales cpu (6->24) / gpu (1->4) / mem (24->96 GB) by num_processes.
    train_job.rqmt["gpu_mem"] = GPU_MEM

    if run_lexicon_search:
        assert recog_rasr_config is not None and recog_lexicon is not None, (
            "run_lexicon_search=True requires recog_rasr_config and recog_lexicon"
        )
        decoder_kwargs = {} if decoder_silence_label is None else {"silence_label": decoder_silence_label}
        default_rasr_decoder_config = RasrDecoderConfig(
            rasr_config_file=recog_rasr_config,
            lexicon=recog_lexicon,
            **decoder_kwargs,
        )

    asr_models_by_epoch = {}
    for epoch in ckpt_list:
        asr_model = prepare_asr_model(
            training_name,
            train_job,
            train_args,
            with_prior=False,
            datasets=train_data,
            get_specific_checkpoint=epoch,
        )
        asr_models_by_epoch[epoch] = asr_model

        if not run_lexicon_search:
            continue

        search_prefix = lexicon_search_root + f"/recog_ep{epoch}"
        asr_model_copy = copy.deepcopy(asr_model)
        asr_model_copy.prior_file = None
        search_jobs, wers = search(
            search_prefix,
            forward_config={
                "num_workers_per_gpu": 0,
                "torch_dataloader_opts": {"num_workers": 0},
            },
            asr_model=asr_model_copy,
            decoder_module="phmm.decoder.rasr_phmm_v1",
            decoder_args={"config": _decoder_config_dict(default_rasr_decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            **returnn,
        )
        # search()/compute_per key by the full "<search_prefix>/<dataset>" name -> dataset -> var.
        wers = {name.split("/")[-1]: wer for name, wer in wers.items()}
        result_kwargs = {"wers": wers}
        if per_lexicon is not None:
            pers = compute_per(search_prefix, search_jobs, dev_dataset_tuples, per_lexicon, hyp_is_phonemes=False)
            result_kwargs["pers"] = {name.split("/")[-1]: per for name, per in pers.items()}
        add_result(
            prefix_name,
            search_type="lexicon",
            model=model_name,
            variant="4gram-word-lm",
            epoch=epoch,
            **result_kwargs,
        )

    return {
        "prefix_name": prefix_name,
        "model_name": model_name,
        "training_name": training_name,
        "train_job": train_job,
        "train_args": train_args,
        "train_data": train_data,
        "dev_dataset_tuples": dev_dataset_tuples,
        "test_dataset_tuples": test_dataset_tuples,
        "asr_models_by_epoch": asr_models_by_epoch,
        "default_returnn": returnn,
        "vocab_size_without_blank": vocab_size_without_blank,
    }


def _decoder_config_dict(cfg) -> Dict[str, Any]:
    """``asdict(cfg)`` but with the optional prior fields DROPPED when no prior is configured.

    The recog ``DecoderConfig`` gained ``prior_file``/``prior_scale`` (default None/0.0). Serializing
    them unconditionally would change every existing recog's hash and invalidate all cached results;
    dropping the defaults keeps the no-prior config byte-identical to the pre-prior layout, so only
    the genuinely prior-enabled recogs get a new hash. No-op for decoder configs without prior fields.
    """
    d = asdict(cfg)
    if d.get("prior_file") is None and not d.get("prior_scale"):
        d.pop("prior_file", None)
        d.pop("prior_scale", None)
    return d


def _select_eval_dataset_tuples(artifacts: Dict[str, Any], dataset_keys: Iterable[str]) -> Dict[str, Any]:
    """Select dev/test dataset tuples from an artifacts dict, preserving the requested key order."""
    dataset_keys = tuple(dataset_keys)
    all_dataset_tuples: Dict[str, Any] = {}
    all_dataset_tuples.update(artifacts.get("dev_dataset_tuples") or {})
    all_dataset_tuples.update(artifacts.get("test_dataset_tuples") or {})
    missing = [key for key in dataset_keys if key not in all_dataset_tuples]
    assert not missing, f"{missing!r} not in {list(all_dataset_tuples)}"
    return {key: all_dataset_tuples[key] for key in dataset_keys}


def compute_phon_prior(artifacts: Dict[str, Any], epoch: int) -> tk.Path:
    """
    Estimate the AM label log-prior (mean softmax posterior over the prior dataset) for one
    checkpoint, using the SAME (LibRASR venv + real-RETURNN) runtime as training/recognition --
    NOT prepare_asr_model's compute_prior, which hardcodes MiniReturnn (wrong for this real-RETURNN
    model). Returns ``prior.txt`` (a [num_labels] vector in +log space; see phmm_zhou.PriorCallback).

    The prior is needed to decode the CTC AM as a (scaled) likelihood instead of a raw posterior:
    cost = -log p(l|x) + prior_scale * log p(l). Omitting it biases the search toward high-marginal
    (frequent) labels -- the substitution-dominated error mode -- and hurts the peaky/blank-dominated
    CTC posteriors far more than the smooth pHMM ones. See [[ctc-vs-phmm-lexical-gap]].
    """
    from .config import get_prior_config
    from .pipeline import compute_prior

    train_args = artifacts["train_args"]
    returnn = artifacts["default_returnn"]
    returnn_config = get_prior_config(
        training_datasets=artifacts["train_data"],
        network_module=train_args["network_module"],
        config={},
        net_args=train_args["net_args"],
        unhashed_net_args=train_args.get("unhashed_net_args", None),
        debug=train_args.get("debug", False),
    )
    prior_file = compute_prior(
        artifacts["training_name"] + f"/ep_{epoch}",
        returnn_config,
        checkpoint=artifacts["train_job"].out_checkpoints[epoch],
        returnn_exe=returnn["returnn_exe"],
        returnn_root=returnn["returnn_root"],
    )
    tk.register_output(artifacts["training_name"] + f"/ep_{epoch}/prior.txt", prior_file)
    return prior_file


def lm_scale_sweep_dev_other(
    *,
    artifacts: Dict[str, Any],
    make_recog,
    lm_scales: Iterable[float],
    epoch: int,
    search_type: str,
    variant_prefix: str,
    returnn: Dict[str, Any],
    per_lexicon=None,
    hyp_is_phonemes: bool = False,
    report_wer: bool = True,
    variant_suffix: str = "",
    scale_label: str = "lm",
    dataset_keys: Iterable[str] = ("dev-other",),
):
    """
    Run a recognition-time **single-scalar sweep on selected eval sets (default dev-other) at a single
    checkpoint**, registering one summary-report row per swept value. Used to tune a recog knob (LM
    scale, or -- with ``scale_label="prior"`` -- the acoustic prior scale) cheaply without re-running
    the full epoch x dataset grid; the model and the training lexicon are never touched (recog-only).

    The swept scalar is named ``{scale_label}{tag}`` in the report variant, so the caller controls
    whether ``lm_scales`` holds LM scales (``scale_label="lm"``, default) or prior scales
    (``scale_label="prior"``); ``make_recog(value)`` decides which knob the value drives.

    Topology-agnostic: the caller supplies ``make_recog(lm_scale) -> dict`` returning the recog
    pieces for that scale, so the same driver tunes the lexicon-search word LM and the lexicon-free
    count phoneme LM alike. The returned dict has:

    * ``"decoder_module"``: the decoder module path (e.g. ``"phmm.decoder.rasr_phmm_v1"``),
    * ``"decoder_config"``: a decoder-config **dataclass instance** (``asdict``-ed into the forward
      config),
    * ``"search_kwargs"`` (optional): extra kwargs for :func:`pipeline.search` (must include
      ``forward_config``; may set ``use_gpu`` / ``mem_rqmt``),
    * ``"gpu_mem"`` (optional): if set, written to each search job's (non-hashed) ``rqmt`` to route a
      GPU recog to the ``gpu_24gb`` partition (see [[gpu-11gb-pool-cuinit-failure]]).

    A scale whose recog config is byte-identical to one already built elsewhere (e.g. the default
    ``lm_scale=1.0`` lexicon-search recog) reuses that job by content hash -- it only adds a report
    row + alias, costing nothing.

    :param artifacts: the dict returned by :func:`train_and_lexicon_search`.
    :param lm_scales: the LM scales to sweep.
    :param epoch: the AM checkpoint to tune at (must be in ``artifacts["asr_models_by_epoch"]``).
    :param search_type: ``"lexicon"`` (-> WER) or ``"lexicon-free"`` (-> PER), for report grouping.
    :param variant_prefix: report-variant stem, e.g. ``"4gram-word-lm"`` / ``"count_o8"``; the scale
        tag ``lm{scale}`` (+ optional ``variant_suffix``) is appended.
    :param returnn: the ``{"returnn_exe", "returnn_root"}`` to run the search with (the count LM needs
        the KenLM venv, so it differs per LM).
    :param per_lexicon: if given, also score an (auxiliary or sole) PER with this phoneme lexicon.
    :param hyp_is_phonemes: passed to :func:`compute_per` (``True`` for lexicon-free phoneme streams).
    :param report_wer: register the WER metric (``True`` for word search; ``False`` for lexicon-free).
    :param variant_suffix: appended after the scale tag, e.g. ``"_beam32_st14p00"`` to match the
        lexicon-free count variants exactly.
    :param dataset_keys: ordered dev/test set selection, e.g. ``("dev-other", "test-other")``.
    """
    asr_model = artifacts["asr_models_by_epoch"].get(epoch)
    if asr_model is None:
        print(f"[lm_scale_sweep_dev_other] no asr_model for epoch {epoch}; skipping sweep")
        return

    prefix_name = artifacts["prefix_name"]
    model_name = artifacts["model_name"]
    eval_dataset_tuples = _select_eval_dataset_tuples(artifacts, dataset_keys)

    section = "lexicon-search" if search_type == "lexicon" else "lexicon-free-search"
    sweep_root = f"{prefix_name}/{section}/{model_name}/lm_scale_sweep"

    for lm_scale in lm_scales:
        recog = make_recog(lm_scale)
        tag = f"{lm_scale:.2f}".replace(".", "p")
        # Include scale_label in the path so a prior sweep (prior0p50) never collides with an LM sweep
        # (lm0p50) under the same sweep_root. The alias/output path is not hashed, so this does not
        # re-key any job (the LM-sweep paths stay lm{tag}, scale_label="lm").
        search_name = f"{sweep_root}/recog_ep{epoch}/{scale_label}{tag}"
        asr_model_copy = copy.deepcopy(asr_model)
        asr_model_copy.prior_file = None
        search_kwargs = dict(recog.get("search_kwargs") or {})
        search_jobs, wers = search(
            search_name,
            asr_model=asr_model_copy,
            decoder_module=recog["decoder_module"],
            decoder_args={"config": _decoder_config_dict(recog["decoder_config"])},
            test_dataset_tuples=eval_dataset_tuples,
            **search_kwargs,
            **returnn,
        )
        gpu_mem = recog.get("gpu_mem")
        if gpu_mem is not None:
            for search_job in search_jobs:
                search_job.rqmt["gpu_mem"] = gpu_mem

        result_kwargs: Dict[str, Any] = {}
        if report_wer:
            result_kwargs["wers"] = {name.split("/")[-1]: wer for name, wer in wers.items()}
        if per_lexicon is not None:
            pers = compute_per(
                search_name,
                search_jobs,
                eval_dataset_tuples,
                per_lexicon,
                hyp_is_phonemes=hyp_is_phonemes,
            )
            result_kwargs["pers"] = {name.split("/")[-1]: per for name, per in pers.items()}
        add_result(
            prefix_name,
            search_type=search_type,
            model=model_name,
            variant=f"{variant_prefix} {scale_label}{tag}{variant_suffix}",
            epoch=epoch,
            **result_kwargs,
        )


def greedy_phon_per_dev_other(
    *,
    artifacts: Dict[str, Any],
    lexicon,
    silence_label: str,
    epoch: int,
    returnn: Dict[str, Any],
    per_lexicon,
    variant_prefix: str = "greedy",
    dataset_key: str = "dev-other",
):
    """
    Greedy (frame-argmax -> collapse repeats -> drop silence/blank) phoneme decode on one dev set at
    one checkpoint, scored by PER. NO LM, NO lexicon constraint, NO beam search -- the cleanest measure
    of the AM's own phoneme quality, used to separate "the CTC AM is genuinely weaker" from "the
    word-constrained search degrades a fine AM". Uses ``phmm.decoder.greedy_phon_v1`` (real-RETURNN
    ``forward_step`` + ``ForwardCallback``: per-frame ``argmax`` -> collapse consecutive duplicates; it
    drops the silence/blank label and any ``[`` / ``<`` tokens, while the EOW ``#`` phonemes are kept)
    and the same ``compute_per(hyp_is_phonemes=True)`` scorer as the lexicon-free path, so the
    greedy PER is directly comparable to the count-LM lexfree PER and (via the same reference
    phonemization) across topologies.

    :param lexicon: phoneme lexicon whose phoneme-inventory ORDER == the AM label order (index 0 =
        silence/blank); this defines index->phoneme for the argmax. CTC: the ``[BLANK]`` lexicon
        (``ctc_lexicon``); pHMM: the ``[SILENCE]`` lexicon (``phmm_lexicon``).
    :param silence_label: the index-0 label to drop (``"[BLANK]"`` for CTC, ``"[SILENCE]"`` for pHMM).
    :param per_lexicon: lexicon to phonemize the WORD reference for PER (words -> EOW phonemes).
    """
    asr_model = artifacts["asr_models_by_epoch"].get(epoch)
    if asr_model is None:
        print(f"[greedy_phon_per_dev_other] no asr_model for epoch {epoch}; skipping")
        return

    prefix_name = artifacts["prefix_name"]
    model_name = artifacts["model_name"]
    dev_dataset_tuples = artifacts["dev_dataset_tuples"]
    assert dataset_key in dev_dataset_tuples, f"{dataset_key!r} not in {list(dev_dataset_tuples)}"
    one_dataset = {dataset_key: dev_dataset_tuples[dataset_key]}

    search_name = f"{prefix_name}/lexicon-free-search/{model_name}/greedy/recog_ep{epoch}"
    asr_model_copy = copy.deepcopy(asr_model)
    asr_model_copy.prior_file = None  # greedy uses raw posteriors -- no prior, no LM
    search_jobs, _wers = search(
        search_name,
        forward_config={},
        asr_model=asr_model_copy,
        decoder_module="phmm.decoder.greedy_phon_v1",
        decoder_args={"config": {"lexicon": lexicon, "silence_label": silence_label}},
        test_dataset_tuples=one_dataset,
        use_gpu=True,
        **returnn,
    )
    for search_job in search_jobs:
        # device="gpu" with no gpu_mem defaults to the flaky gpu_11gb pool (cuInit fails); pin gpu_24gb.
        search_job.rqmt["gpu_mem"] = 24
    pers = compute_per(search_name, search_jobs, one_dataset, per_lexicon, hyp_is_phonemes=True)
    add_result(
        prefix_name,
        search_type="lexicon-free",
        model=model_name,
        variant=variant_prefix,
        epoch=epoch,
        pers={name.split("/")[-1]: per for name, per in pers.items()},
    )
