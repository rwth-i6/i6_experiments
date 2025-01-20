import functools
from typing import Any, Dict, List, Optional, Union
import copy

import numpy as np

import i6_core.returnn as returnn
import i6_experiments.users.jxu.experiments.ctc.tedlium2.configs.configs_helper as configs_helper
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant

# ********** Constant values **********

num_outputs = 79
num_subepochs = 250


# ********** Settings **********

def get_returnn_config(
        network: Optional[Dict] = None,
        *,
        target: Optional[str] = "classes",
        num_inputs: Optional[int] = None,
        num_outputs: Optional[int] = None,
        python_prolog: Optional[Union[List, Dict]] = None,
        extern_data_config: bool = False,
        extra_python: Optional[List] = None,
        extra_config: Optional[Dict] = None,
        hash_full_python_code: bool = False,
        **kwargs,
) -> returnn.ReturnnConfig:
    python_prolog = python_prolog or ["import numpy as np"]
    extra_python = extra_python or []
    config_dict: dict[str, Any] = {"target": target}
    if num_inputs is not None:
        config_dict["num_inputs"] = num_inputs
    if num_outputs is not None:
        config_dict["num_outputs"] = {target: num_outputs}
    if extern_data_config:
        config_dict.update(
            configs_helper.get_extern_data_config(num_inputs=1, num_outputs=num_outputs, target=target,
                                                  **kwargs)
        )
    config_dict.update(configs_helper.get_base_config())

    if network:
        config_dict.update({"network:": network})

    lrate_config = configs_helper.get_oclr_config(**kwargs)
    config_dict.update(lrate_config)

    config_dict.update(configs_helper.get_base_regularization_config(**kwargs))

    if extra_config:
        config_dict.update(extra_config)

    post_config_dict = {}
    post_config_dict.update(configs_helper.get_base_post_config(**kwargs))

    return returnn.ReturnnConfig(
        config=config_dict,
        post_config=post_config_dict,
        hash_full_python_code=hash_full_python_code,
        python_prolog=python_prolog,
        python_epilog=extra_python,
        pprint_kwargs={"sort_dicts": False},
    )

def get_serializer(model_config, variant: ConfigVariant, in_dim: int = 1) -> Collection:
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.dynamic_encoder_size.orthogonal_softmax.joint_train_conformer_orthogonal_softmax_layer_wise import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(train_data_config: dict, dev_data_config: dict, peak_lr: float, batch_size: int) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.dynamic_encoder_size.orthogonal_softmax.joint_train_conformer_orthogonal_softmax_layer_wise import \
        get_default_config_v1 as get_train_config
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.dynamic_encoder_size.orthogonal_softmax.joint_train_conformer_orthogonal_softmax_layer_wise import \
        get_default_config_v1 as get_recog_config

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    recog_extra_config = copy.deepcopy(extra_config)
    recog_extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}

    config_partial = functools.partial(
        get_returnn_config,
        num_epochs=num_subepochs,
        num_inputs=80,
        num_outputs=num_outputs,
        target="targets",
        extern_data_config=True,
        grad_noise=0.0,
        grad_clip=0.0,
        cycle_epoch=110,
        lr_1= peak_lr / 100,
        lr_2 = peak_lr / 10,
        peak_lr=peak_lr,
        final_lr=1e-08,
        batch_size=batch_size*160,
        extra_config=extra_config,
    )

    def get_returnn_configs(train_config, recog_config):
        return ReturnnConfigs(
            train_config=config_partial(
                extra_python=[get_serializer(train_config, ConfigVariant.TRAIN)],
                extra_config=extra_config),
            prior_config=config_partial(extra_python=[get_serializer(recog_config, ConfigVariant.PRIOR)],
                                        extra_config=extra_config),
            recog_configs={
                "recog": config_partial(extra_python=[get_serializer(recog_config, ConfigVariant.RECOG)],
                                        extra_config=recog_extra_config)},
        )

    experiments = {}

    for num_iters in [1]:
        for layer_dropout_stage_2 in [0.1, 0.3]:
            for initial_tau, min_tau in [(1, 1)]:
                for softmax_constraint_norm in ["L2_norm_sqrt"]:
                    for softmax_constraint_loss_scale in ["linear_increase"]:
                        for aux_loss_scale in ["focal_loss"]:
                            for dropout, final_dropout in [(0.2, 0.2)]:
                                for max_softmax_constraint_loss_scale, softmax_constraint_warmup_steps in [(1, 150)]:
                                    layer_dropout_kwargs = {"layer_dropout_stage_1": 0,
                                                            "layer_dropout_stage_2": layer_dropout_stage_2}
                                    aux_loss_scales = {'0.39': aux_loss_scale, '0.51': aux_loss_scale,'0.63': aux_loss_scale, '0.81': aux_loss_scale, "1": 1}
                                    softmax_kwargs = {"softmax_constraint_norm": softmax_constraint_norm,
                                                      "initial_tau": initial_tau, "min_tau": min_tau, "tau_annealing": 0.999992,
                                                      "softmax_constraint_loss_scale": softmax_constraint_loss_scale, "new_train": True}
                                    if softmax_constraint_loss_scale == "linear_increase":
                                        softmax_kwargs[
                                            "max_softmax_constraint_loss_scale"] = max_softmax_constraint_loss_scale
                                        softmax_kwargs[
                                            "softmax_constraint_warmup_steps"] = softmax_constraint_warmup_steps * 1500

                                    params_kwargs = {"num_params": ([8.27] + [3.21] + [4.18] + [8.27]) * 12,
                                                     "rest_params": 110,
                                                     "total_params": 400}

                                    network_args = {"time_max_mask_per_n_frames": 25,
                                                    "freq_max_num_masks": 16,
                                                    "final_dropout": final_dropout,
                                                    "dropout": dropout,
                                                    "softmax_kwargs": softmax_kwargs,
                                                    "layer_dropout_kwargs":layer_dropout_kwargs,
                                                    "recog_param_pct": 1,
                                                    "aux_loss_scales": aux_loss_scales,
                                                    "pct_params_set": [0.39, 0.51, 0.63, 0.81, 1],
                                                    "stage_1_global_steps": 150*1500,
                                                    "params_kwargs":params_kwargs}

                                    train_config = get_train_config(num_inputs=80, num_outputs=num_outputs,
                                                                    network_args=network_args)
                                    recog_network_args = copy.deepcopy(network_args)

                                    for pct in [0.39, 0.51, 0.63, 0.81, 1]:
                                        recog_network_args = copy.deepcopy(network_args)
                                        recog_network_args["recog_param_pct"] = pct
                                        recog_config = get_recog_config(num_inputs=80,
                                                                        num_outputs=num_outputs,
                                                                        network_args=recog_network_args)
                                        if softmax_constraint_loss_scale == "linear_increase":
                                            exp_name = f"no_tau_pct_0_39_layerdrop_0_{layer_dropout_stage_2}_final_dropout_{final_dropout}_softmax_{softmax_constraint_norm}_scale_{softmax_constraint_loss_scale}_{max_softmax_constraint_loss_scale}_{softmax_constraint_warmup_steps}_aux_loss_scale_{aux_loss_scale}_{pct}_model"
                                        else:
                                            exp_name = f"final_dropout_{final_dropout}_softmax_{softmax_constraint_norm}_scale_{softmax_constraint_loss_scale}_aux_loss_scale_{aux_loss_scale}_{pct}_model"
                                        exp_name = exp_name.replace(".", "_")
                                        experiments[exp_name] = get_returnn_configs(
                                            train_config,
                                            recog_config)
    return experiments
