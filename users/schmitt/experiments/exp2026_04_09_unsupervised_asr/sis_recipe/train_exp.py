import copy
from typing import cast, Dict, Sequence, Optional, Union, Tuple, List, Iterator, Any, Callable
import inspect
from functools import cache

from sisyphus import tk

from i6_core.serialization import Collection
from i6_core.returnn.config import CodeWrapper, ReturnnConfig

from . import learning_rate_configs
from .tune_eval import eval_model, eval_model_rasr
from .analysis import analyze_encoder_states
from .ppl import compute_ppl
from .pipeline import training
from .default_tools import RETURNN_EXE, RETURNN_ROOT
from ..models.recognition.discrete_audio_aed.beam_search import DecoderConfig

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


@cache
def _get_existing_lr_configs():
    existing_lr_configs = inspect.getmembers(learning_rate_configs, predicate=inspect.isfunction)
    existing_lr_configs = list(map(lambda x: x[0], existing_lr_configs)) + ["cosine_annealing"]
    existing_lr_configs = [
        config
        for config in existing_lr_configs
        if config not in ("_get_piecewise_lr_function", "cache", "get_lr_config")
    ]
    return existing_lr_configs


def run_train(
    training_name: str,
    config: Dict,
    train_data,
    keep_epochs: Optional[List[int]] = None,
    additional_configs: Optional[List[ReturnnConfig]] = None,
    cleanup_old_models: Optional[Dict[str, Any]] = None,
):
    num_epochs = config["training"].pop("__num_epochs")
    network_module = config.pop("__network_module")
    train_step_module = config.pop("__train_step_module")
    lr_opts = config["training"].pop("__lr_opts")

    if keep_epochs is None:
        keep_epochs = [num_epochs]

    if cleanup_old_models is None:
        cleanup_old_models = {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": keep_epochs,
        }

    if additional_configs is None:
        additional_configs = []

    lr_type = lr_opts.pop("type")
    existing_lr_configs = _get_existing_lr_configs()
    if lr_type in existing_lr_configs:
        if lr_type == "cosine_annealing":
            # legacy
            func_name = "linear_warmup_cosine_annealing"
        else:
            func_name = lr_type
        additional_configs.append(
            learning_rate_configs.get_lr_config(
                func_name=func_name,
                num_epochs=num_epochs,
                **lr_opts,
            )
        )
    else:
        raise ValueError(f"unknown lr type: {lr_type}")

    # batch size, adamw, speed pert, gradient clip,
    train_args = {
        "config": {**config["training"], **config["general"]},
        "post_config": {
            "cleanup_old_models": cleanup_old_models,
            **(config.get("train_post_config", {})),
        },
        "python_prolog": [
            "import os",
            'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"',  # to tackle OOM
        ],
        "network_module": network_module,
        "train_step_module": train_step_module,
        "net_args": config["model_args"],
        "train_args": config["train_args"],
        "rqmt": config.pop("train_rqmt", {}),
    }
    train_job = training(
        training_name,
        train_data,
        train_args,
        num_epochs=num_epochs,
        additional_configs=additional_configs,
        **default_returnn,
    )

    return train_job, train_args


def run_eval(
    training_name: str,
    train_job,
    train_args,
    config: Dict,
    train_data,
    test_data_dict: Dict[str, Tuple],
    keep_epochs: Optional[List[int]] = None,
    recog_name: str = "recog",
    network_module: Optional[str] = None,
    extra_forward_config: Optional[ReturnnConfig] = None,
    decoder_config: DecoderConfig = DecoderConfig(
        beam_size=12,
    ),
    recog_model_args: Optional[Dict] = None,
    main_eval_measure_key: str = "dev",
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
    input_modality: str = "audio",
    output_modality: str = "text",
    mask_input: bool = False,
    masking_opts: Optional[Dict[str, Any]] = None,
    expansion_opts: Optional[Dict[str, Any]] = None,
):
    forward_step_module = config.pop("__forward_step_module")
    callback_module = config.pop("__callback_module")

    # don't mutate the caller's train_args (run_eval may be called multiple times, e.g. for several
    # recog variants), only override on a local copy when needed.
    if network_module is not None or recog_model_args is not None:
        train_args = copy.deepcopy(train_args)
        if network_module is not None:
            train_args["network_module"] = network_module
        if recog_model_args is not None:
            train_args["net_args"] = recog_model_args
    eval_model(
        config={**config["general"], **config.get("recog", {})},
        recog_name=recog_name,
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        base_decoder_config=decoder_config,
        decoder_module=forward_step_module,
        callback_module=callback_module,
        checkpoints=keep_epochs,
        test_data_dict=test_data_dict,
        extra_forward_config=extra_forward_config,
        main_eval_measure_key=main_eval_measure_key,
        rqmt=config.get("recog_rqmt", None),
        recog_post_proc_funcs=recog_post_proc_funcs,
        input_modality=input_modality,
        output_modality=output_modality,
        mask_input=mask_input,
        masking_opts=masking_opts,
        expansion_opts=expansion_opts,
    )


def run_rasr_eval(
    training_name: str,
    train_job,
    train_args,
    config: Dict,
    train_data,
    test_data_dict: Dict[str, Tuple],
    recog_opts: Dict,
    keep_epochs: Optional[List[int]] = None,
    recog_name: str = "recog",
    network_module: Optional[str] = None,
    extra_forward_config: Optional[ReturnnConfig] = None,
    decoder_config: Optional[DecoderConfig] = None,
    recog_model_args: Optional[Dict] = None,
    main_eval_measure_key: str = "dev",
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
    input_modality: str = "audio",
    output_modality: str = "text",
    mask_input: bool = False,
    masking_opts: Optional[Dict[str, Any]] = None,
):
    forward_step_module = config.pop("__rasr_forward_step_module")
    callback_module = config.pop("__rasr_callback_module")
    export_forward_step = config.pop("__onnx_export_forward_step_module")

    # don't mutate the caller's train_args (run_eval may be called multiple times, e.g. for several
    # recog variants), only override on a local copy when needed.
    if network_module is not None or recog_model_args is not None:
        train_args = copy.deepcopy(train_args)
        if network_module is not None:
            train_args["network_module"] = network_module
        if recog_model_args is not None:
            train_args["net_args"] = recog_model_args
    eval_model_rasr(
        recog_config={**config["general"], **config.get("rasr_recog", {})},
        onnx_config={**config["general"]},
        recog_name=f"{recog_name}_rasr",
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        base_decoder_config=decoder_config,
        decoder_module=forward_step_module,
        callback_module=callback_module,
        checkpoints=keep_epochs,
        test_data_dict=test_data_dict,
        extra_forward_config=extra_forward_config,
        main_eval_measure_key=main_eval_measure_key,
        rqmt=config.get("recog_rqmt", None),
        recog_post_proc_funcs=recog_post_proc_funcs,
        input_modality=input_modality,
        output_modality=output_modality,
        mask_input=mask_input,
        masking_opts=masking_opts,
        recog_opts=recog_opts,
        export_forward_step=export_forward_step,
    )


def run_analysis(
    training_name: str,
    train_job,
    train_args,
    config: Dict,
    train_data,
    test_data_dict: Dict[str, Tuple],
    checkpoints: List[Union[int, str]],
    analysis_name: str = "encoder_pca",
    extra_forward_config: Optional[ReturnnConfig] = None,
    **analysis_kwargs,
):
    analyze_encoder_states(
        config={**config["general"], **config.get("recog", {})},
        training_name=training_name,
        analysis_name=analysis_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        test_data_dict=test_data_dict,
        checkpoints=checkpoints,
        extra_forward_config=extra_forward_config,
        **analysis_kwargs,
    )


def run_ppl(
    training_name: str,
    train_job,
    train_args,
    config: Dict,
    train_data,
    test_data_dict: Dict[str, Tuple],
    checkpoints: List[Union[int, str]],
    ppl_name: str = "ppl",
    extra_forward_config: Optional[ReturnnConfig] = None,
    **ppl_kwargs,
):
    # Like run_analysis, the PPL forward step/callback are fixed (compute_ppl's PPL_* defaults), NOT
    # taken from the config's __forward_step_module/__callback_module -- those point at the beam-search
    # recog step for the AED configs, which is unrelated to perplexity scoring.
    compute_ppl(
        config={**config["general"], **config.get("recog", {})},
        training_name=training_name,
        analysis_name=ppl_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        test_data_dict=test_data_dict,
        checkpoints=checkpoints,
        extra_forward_config=extra_forward_config,
        **ppl_kwargs,
    )


def run_experiment(
    training_name: str,
    config: Dict,
    train_data,
    test_data_dict: Dict[str, Tuple],
    keep_epochs: Optional[List[int]] = None,
    recog_name: str = "recog",
    network_module_recog: Optional[str] = None,
    extra_forward_config: Optional[ReturnnConfig] = None,
    decoder_config: DecoderConfig = DecoderConfig(
        beam_size=12,
    ),
    recog_model_args: Optional[Dict] = None,
    additional_configs: Optional[List[ReturnnConfig]] = None,
    main_eval_measure_key: str = "dev-other",
    cleanup_old_models: Optional[Dict[str, Any]] = None,
    skip_eval: bool = False,
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
    analysis_opts: Optional[Dict[str, Any]] = None,
    ppl_opts: Optional[Dict[str, Any]] = None,
    recog_variants: Optional[List[Dict[str, Any]]] = None,
    rasr_recog_opts: Optional[Dict] = None,
):
    """
    :param skip_eval: skip the standard (audio->text ASR) recognition + scoring.
    :param ppl_opts: if given, additionally run the perplexity forward job (models.scoring.ppl) via
        :func:`run_ppl`. A kwargs dict for :func:`run_ppl`, e.g. ``{"checkpoints": [1000]}`` for the
        phoneme LM (decoder-only), or ``{"checkpoints": [1000], "input_modality": "audio"}`` to score
        the conditional (audio->text) PPL of an AED model -- for the AED path the scoring dataset must
        expose both audio + text (like the encoder-PCA analysis dataset). May include a
        ``"test_data_dict"`` key to score PPL on a *separate* dataset (overriding the recognition
        ``test_data_dict`` for the PPL job only) -- e.g. a wo-silence reference matching a wo-sil model
        while recognition keeps the with-silence test set.
    :param recog_variants: extra recognition variants run in addition to the standard one. Each is a
        kwargs dict for :func:`run_eval`, e.g. ``{"recog_name": "recon_audio", "input_modality":
        "audio", "output_modality": "audio", "mask_input": True, "masking_opts": {...}}`` to score
        same-modality reconstruction. Use a distinct ``recog_name`` per variant. A variant may also
        set ``keep_epochs`` to recognize only specific checkpoints (else the experiment's
        ``keep_epochs`` are used).
    """
    train_job, train_args = run_train(
        training_name=training_name,
        config=copy.deepcopy(config),
        train_data=train_data,
        keep_epochs=keep_epochs,
        additional_configs=additional_configs,
        cleanup_old_models=cleanup_old_models,
    )

    if analysis_opts is not None:
        run_analysis(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            **analysis_opts,
        )

    if ppl_opts is not None:
        # the PPL job may score on its own dataset (e.g. a wo-silence reference matching a wo-sil
        # model, while recognition still uses the with-silence test_data_dict). If ppl_opts provides
        # a "test_data_dict", it overrides the recognition test_data_dict for the PPL job only.
        ppl_opts = dict(ppl_opts)
        ppl_test_data_dict = ppl_opts.pop("test_data_dict", test_data_dict)
        run_ppl(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=ppl_test_data_dict,
            **ppl_opts,
        )

    if not skip_eval:
        run_eval(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_epochs,
            recog_name=recog_name,
            network_module=network_module_recog,
            extra_forward_config=extra_forward_config,
            decoder_config=decoder_config,
            recog_model_args=recog_model_args,
            main_eval_measure_key=main_eval_measure_key,
            recog_post_proc_funcs=recog_post_proc_funcs,
        )

    if rasr_recog_opts is not None:
        run_rasr_eval(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_epochs,
            recog_name=recog_name,
            network_module=network_module_recog,
            extra_forward_config=extra_forward_config,
            decoder_config=None,
            recog_model_args=recog_model_args,
            main_eval_measure_key=main_eval_measure_key,
            recog_post_proc_funcs=recog_post_proc_funcs,
            recog_opts=rasr_recog_opts,
        )

    # additional recognition variants (e.g. same-modality reconstruction), each with its own
    # recog_name and modality/masking options. These run regardless of skip_eval. A variant may
    # override `keep_epochs` to recognize only specific checkpoints (else the experiment's
    # keep_epochs are used).
    for variant in recog_variants or []:
        variant = dict(variant)
        variant_keep_epochs = variant.pop("keep_epochs", keep_epochs)
        run_eval(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=variant_keep_epochs,
            network_module=network_module_recog,
            extra_forward_config=extra_forward_config,
            decoder_config=decoder_config,
            recog_model_args=recog_model_args,
            main_eval_measure_key=main_eval_measure_key,
            recog_post_proc_funcs=recog_post_proc_funcs,
            **variant,
        )

    return train_job
