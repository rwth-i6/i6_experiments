import copy
from typing import cast, Dict, Sequence, Optional, Union, Tuple, List, Iterator, Any, Callable
import inspect
from functools import cache

from sisyphus import tk

from i6_core.serialization import Collection
from i6_core.returnn.config import CodeWrapper, ReturnnConfig

from . import learning_rate_configs
from .tune_eval import eval_model
from i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.tune_eval import eval_model_rasr
from .analysis import analyze_encoder_states
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
    if network_module is not None and not network_module.startswith("i6_experiments"):
        from ..models import PACKAGE
        network_module = f"{PACKAGE}.{network_module}"
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
            "import sys",
            'os.environ["PYTHONPATH"] = ":".join(sys.path)',  # Export to child spawn workers
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
):
    forward_step_module = config.pop("__forward_step_module")
    callback_module = config.pop("__callback_module")

    python_prolog_env_patch = [
        "import os",
        "import sys",
        'os.environ["PYTHONPATH"] = ":".join(sys.path)',
        'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"',
    ]
    if extra_forward_config is None:
        extra_forward_config = ReturnnConfig(config={}, python_prolog=python_prolog_env_patch)
    else:
        extra_forward_config.python_prolog = python_prolog_env_patch + (extra_forward_config.python_prolog or [])

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
    )

 


def run_encoder_embedding_visualization(
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
): 

    analyze_encoder_states(
        config={**config["general"], **config.get("recog", {})},
        analysis_name=recog_name,
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        test_data_dict=test_data_dict,
        checkpoints=keep_epochs,
        extra_forward_config=extra_forward_config,
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
    rasr_recog_opts: Optional[Dict] = None,
    vis_epochs: Optional[List[int]] = None,
):
    train_job, train_args = run_train(
        training_name=training_name,
        config=copy.deepcopy(config),
        train_data=train_data,
        keep_epochs=keep_epochs,
        additional_configs=additional_configs,
        cleanup_old_models=cleanup_old_models,
    )

    if skip_eval:
        return train_job

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

    if vis_epochs is not None:
        run_encoder_embedding_visualization(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=vis_epochs,
            recog_name="vis_encoder_pca",
            network_module=network_module_recog,
            extra_forward_config=extra_forward_config,
            decoder_config=decoder_config,
            recog_model_args=recog_model_args,
            main_eval_measure_key=main_eval_measure_key,
            recog_post_proc_funcs=recog_post_proc_funcs,
        )

    return train_job
