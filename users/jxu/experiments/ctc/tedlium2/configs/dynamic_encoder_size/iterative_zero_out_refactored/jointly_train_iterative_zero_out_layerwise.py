import functools
from typing import Any, Dict, List, Optional, Union
import copy

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
            configs_helper.get_extern_data_config(num_inputs=num_inputs, num_outputs=num_outputs, target=target,
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
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.dynamic_encoder_size.zeroout_with_new_i6_models.joint_train_two_model_iterative_zero_out_modwise import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(train_data_config: dict, dev_data_config: dict, peak_lr: float) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.dynamic_encoder_size.zeroout_with_new_i6_models.joint_train_two_model_iterative_zero_out_modwise import \
        get_default_config_v1 as get_train_config
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.dynamic_encoder_size.zeroout_with_new_i6_models.joint_train_two_model_iterative_zero_out_modwise import \
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
        num_inputs=50,
        num_outputs=num_outputs,
        target="targets",
        extern_data_config=True,
        grad_noise=0.0,
        grad_clip=0.0,
        cycle_epoch=110,
        initial_lr=peak_lr / 100,
        peak_lr=peak_lr,
        final_lr=1e-08,
        batch_size=15000,
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

    # ----------------------------- gumbel_scale 0.05 gumble_top_k_dropout_0_3 init tau 0.5 k annealing every 20 sub-epoch ---------------------------------------

    experiments = {}
    for num_steps_per_iter in [[45 * 1400] + [15 * 1400] * 7]:
        for num_zeroout_elements_per_iter in [[3,6,9,12,15,18,21,24]]:
            for layer_dropout_stage_1, layer_dropout_stage_2 in [(0, 0.3)]:
                layer_dropout_kwargs = {"layer_dropout_stage_1": layer_dropout_stage_1,
                                        "layer_dropout_stage_2": layer_dropout_stage_2}
                zero_out_kwargs = {"num_steps_per_iter":num_steps_per_iter, "num_zeroout_elements_per_iter":num_zeroout_elements_per_iter, "zeroout_val": -5}
                network_args = {"layer_dropout_kwargs": layer_dropout_kwargs,
                                "zero_out_kwargs": zero_out_kwargs, "num_layers_set": [24, 48],
                                "layer_gate_activation":"sigmoid", "recog_num_layers": 48}

                train_config = get_train_config(num_inputs=50, num_outputs=num_outputs,
                                                network_args=network_args)
                recog_network_args = copy.deepcopy(network_args)
                recog_network_args["recog_num_layers"] = 48
                num_recog_mods_48_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                                  network_args=recog_network_args)
                recog_network_args["recog_num_layers"] = 24
                num_recog_mods_24_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                                  network_args=recog_network_args)

                num_recog_mods_48_experiment_name = f"two_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_48_peak_lr_{peak_lr}"
                num_recog_mods_48_experiment_name = num_recog_mods_48_experiment_name.replace(".", "_")
                experiments[num_recog_mods_48_experiment_name] = get_returnn_configs(train_config,
                                                                                     num_recog_mods_48_recog_config)

                num_recog_mods_24_experiment_name = f"two_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_24_peak_lr_{peak_lr}"
                num_recog_mods_24_experiment_name = num_recog_mods_24_experiment_name.replace(".", "_")
                experiments[num_recog_mods_24_experiment_name] = get_returnn_configs(train_config,
                                                                                     num_recog_mods_24_recog_config)

    for num_steps_per_iter in [[42000,25200,25200,25200,25200,25200,25200,25200]]:
        for num_zeroout_elements_per_iter in [[4,8,12,16,20,24,28,32]]:
            layer_dropout_kwargs = {"layer_dropout_stage_1": layer_dropout_stage_1,
                                    "layer_dropout_stage_2": layer_dropout_stage_2}
            zero_out_kwargs = {"num_steps_per_iter": num_steps_per_iter,
                               "num_zeroout_elements_per_iter": num_zeroout_elements_per_iter, "zeroout_val": -5}

            network_args = {"layer_dropout_kwargs": layer_dropout_kwargs,
                            "zero_out_kwargs": zero_out_kwargs, "recog_num_layers": 48, "num_layers_set": [16, 32, 48],
                            "layer_gate_activation":"sigmoid"}

            train_config = get_train_config(num_inputs=50, num_outputs=num_outputs,
                                            network_args=network_args)
            recog_network_args = copy.deepcopy(network_args)
            recog_network_args["recog_num_layers"] = 48
            num_recog_mods_48_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                              network_args=recog_network_args)

            recog_network_args["recog_num_layers"] = 32
            num_recog_mods_32_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                              network_args=recog_network_args)

            recog_network_args["recog_num_layers"] = 16
            num_recog_mods_16_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                              network_args=recog_network_args)

            num_recog_mods_48_experiment_name = f"three_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_48_peak_lr_{peak_lr}"
            num_recog_mods_48_experiment_name = num_recog_mods_48_experiment_name.replace(".", "_")
            experiments[num_recog_mods_48_experiment_name] = get_returnn_configs(train_config,
                                                                                 num_recog_mods_48_recog_config)

            num_recog_mods_32_experiment_name = f"three_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_32_peak_lr_{peak_lr}"
            num_recog_mods_32_experiment_name = num_recog_mods_32_experiment_name.replace(".", "_")
            experiments[num_recog_mods_32_experiment_name] = get_returnn_configs(train_config,
                                                                                 num_recog_mods_32_recog_config)

            num_recog_mods_16_experiment_name = f"three_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_16_peak_lr_{peak_lr}"
            num_recog_mods_16_experiment_name = num_recog_mods_16_experiment_name.replace(".", "_")
            experiments[num_recog_mods_16_experiment_name] = get_returnn_configs(train_config,
                                                                                 num_recog_mods_16_recog_config)


    for num_steps_per_iter in [[10*1400]+[4*1400]*35]:
        for num_zeroout_elements_per_iter in [list(range(1,37))]:
            for layer_dropout_stage_1, layer_dropout_stage_2 in [(0, 0.3)]:
                zero_out_kwargs = {"num_steps_per_iter": num_steps_per_iter,
                                   "num_zeroout_elements_per_iter": num_zeroout_elements_per_iter, "zeroout_val": -5}
                layer_dropout_kwargs = {"layer_dropout_stage_1": layer_dropout_stage_1,
                                        "layer_dropout_stage_2": layer_dropout_stage_2}
                network_args = {"layer_dropout_kwargs": layer_dropout_kwargs,
                                "zero_out_kwargs": zero_out_kwargs, "recog_num_layers": 48, "num_layers_set": [12, 24, 36, 48],
                                "layer_gate_activation":"sigmoid"}

                train_config = get_train_config(num_inputs=50, num_outputs=num_outputs,
                                                network_args=network_args)
                recog_network_args = copy.deepcopy(network_args)
                recog_network_args["recog_num_layers"] = 48
                num_recog_mods_48_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                                   network_args=recog_network_args)

                recog_network_args["recog_num_layers"] = 36
                num_recog_mods_36_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                                   network_args=recog_network_args)

                recog_network_args["recog_num_layers"] = 24
                num_recog_mods_24_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                                   network_args=recog_network_args)

                recog_network_args["recog_num_layers"] = 12
                num_recog_mods_12_recog_config = get_recog_config(num_inputs=50, num_outputs=num_outputs,
                                                                  network_args=recog_network_args)

                num_recog_mods_48_experiment_name = f"four_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_48_peak_lr_{peak_lr}"
                num_recog_mods_48_experiment_name = num_recog_mods_48_experiment_name.replace(".", "_")
                experiments[num_recog_mods_48_experiment_name] = get_returnn_configs(train_config, num_recog_mods_48_recog_config)

                num_recog_mods_36_experiment_name = f"four_models_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_36_peak_lr_{peak_lr}"
                num_recog_mods_36_experiment_name = num_recog_mods_36_experiment_name.replace(".", "_")
                experiments[num_recog_mods_36_experiment_name] = get_returnn_configs(train_config, num_recog_mods_36_recog_config)

                num_recog_mods_24_experiment_name = f"four_models_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_24_peak_lr_{peak_lr}"
                num_recog_mods_24_experiment_name = num_recog_mods_24_experiment_name.replace(".", "_")
                experiments[num_recog_mods_24_experiment_name] = get_returnn_configs(train_config, num_recog_mods_24_recog_config)

                num_recog_mods_12_experiment_name = f"four_models_dropout_{layer_dropout_stage_1}_{layer_dropout_stage_2}_recog_num_mods_12_peak_lr_{peak_lr}"
                num_recog_mods_12_experiment_name = num_recog_mods_12_experiment_name.replace(".", "_")
                experiments[num_recog_mods_12_experiment_name] = get_returnn_configs(train_config, num_recog_mods_12_recog_config)

    return experiments
