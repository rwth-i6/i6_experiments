import copy
from typing import Any, Dict, Optional

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection as TorchCollection
from i6_experiments.common.setups.serialization import Import

from .data.common import TrainingDatasets
from .serializer import PACKAGE, serialize_forward, serialize_training


def get_training_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
    use_speed_perturbation: bool = False,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    base_config = {
        "cleanup_old_models": {"keep_last_n": 4, "keep_best_n": 4},
        "train": copy.deepcopy(training_datasets.train.as_returnn_opts()),
        "dev": training_datasets.cv.as_returnn_opts(),
        "eval_datasets": {"devtrain": training_datasets.devtrain.as_returnn_opts()},
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config = {"stop_on_nonfinite_train_score": True, "backend": "torch", **copy.deepcopy(post_config or {})}

    python_prolog_objects = []
    if use_speed_perturbation:
        python_prolog_objects.append(
            Import(
                code_object_path=PACKAGE + ".extra_code.speed_perturbation.legacy_speed_perturbation",
                unhashed_package_root=PACKAGE,
            )
        )
        config["train"]["datasets"]["zip_dataset"]["audio"]["pre_process"] = CodeWrapper("legacy_speed_perturbation")

    python_prolog = [TorchCollection(python_prolog_objects)] if python_prolog_objects else None
    return ReturnnConfig(
        config=config,
        post_config=post_config,
        python_prolog=python_prolog,
        python_epilog=[
            serialize_training(
                network_module=network_module,
                net_args=net_args,
                unhashed_net_args=unhashed_net_args,
                include_native_ops=include_native_ops,
                debug=debug,
            )
        ],
    )


def get_prior_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
):
    base_config = {
        "num_workers_per_gpu": 2,
        "batch_size": 500 * 16000,
        "max_seqs": 240,
        "forward": copy.deepcopy(training_datasets.prior.as_returnn_opts()),
    }
    config = {**base_config, **copy.deepcopy(config)}
    return ReturnnConfig(
        config=config,
        post_config={"backend": "torch"},
        python_prolog=[TorchCollection([])],
        python_epilog=[
            serialize_forward(
                network_module=network_module,
                net_args=net_args,
                unhashed_net_args=unhashed_net_args,
                forward_step_name="prior",
                debug=debug,
            )
        ],
    )


def get_forward_config(
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    decoder: str,
    decoder_args: Dict[str, Any],
    unhashed_decoder_args: Optional[Dict[str, Any]] = None,
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
):
    base_config = {"batch_size": 100 * 16000, "max_seqs": 200}
    config = {**base_config, **copy.deepcopy(config)}
    return ReturnnConfig(
        config=config,
        post_config={"backend": "torch"},
        python_prolog=[TorchCollection([])],
        python_epilog=[
            serialize_forward(
                network_module=network_module,
                net_args=net_args,
                unhashed_net_args=unhashed_net_args,
                forward_module=decoder,
                forward_step_name="forward",
                forward_init_args=decoder_args,
                unhashed_forward_init_args=unhashed_decoder_args,
                debug=debug,
            )
        ],
    )
