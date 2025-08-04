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
num_subepochs = 600


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
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_conformer.adapt_based_on_gradient_ranking_finer_granul_double_and_prune import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(train_data_config: dict, dev_data_config: dict, peak_lr: float) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_conformer.adapt_based_on_gradient_ranking_finer_granul_double_and_prune import \
        get_default_config_v1

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
        cycle_epoch=(num_subepochs * 9) // 20,
        lr_1=peak_lr / 100,
        lr_2=peak_lr / 100,
        peak_lr=peak_lr,
        final_lr=1e-08,
        batch_size=15000 * 160,
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

    dict_recog_component_dist = {
        "double_prune_total_num_components_216_adjust_dropout_True_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [2, 4, 4, 0, 0, 0, 0, 4, 4, 4, 6, 4],
             'conv': [2, 2, 6, 4, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_216_noise_var_0_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [2, 4, 3, 3, 0, 0, 1, 4, 4, 4, 4, 4],
             'conv': [2, 2, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8], 'mhsa': [7, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_216_noise_var_0.1_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [1, 4, 4, 1, 0, 0, 1, 4, 4, 4, 4, 4],
             'conv': [2, 2, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8], 'mhsa': [6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 5, 0]},
        "double_prune_total_num_components_216_noise_var_0.0001_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [2, 4, 4, 0, 0, 0, 1, 4, 4, 4, 5, 4],
             'conv': [2, 2, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},

        "double_prune_total_num_components_1224_adjust_dropout_True_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [21, 32, 32, 32, 31, 31, 32, 32, 32, 32, 64, 64],
             'ff2': [15, 32, 21, 9, 5, 3, 16, 30, 32, 32, 44, 32],
             'conv': [13, 21, 36, 41, 45, 47, 49, 52, 63, 64, 64, 64], 'mhsa': [6, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_1224_noise_var_0_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [31, 33, 32, 32, 32, 30, 32, 32, 32, 32, 53, 64],
             'ff2': [23, 32, 25, 11, 0, 0, 5, 24, 32, 32, 34, 32],
             'conv': [11, 24, 44, 45, 47, 48, 49, 56, 62, 64, 64, 64], 'mhsa': [9, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_1224_noise_var_0.1_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [32, 33, 32, 32, 32, 32, 32, 32, 32, 32, 58, 64],
             'ff2': [23, 32, 28, 2, 1, 4, 2, 25, 32, 32, 38, 32],
             'conv': [13, 23, 43, 44, 42, 45, 49, 51, 60, 64, 64, 64], 'mhsa': [8, 12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_1224_noise_var_0.0001_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [24, 32, 32, 32, 29, 26, 32, 32, 32, 32, 61, 64],
             'ff2': [18, 32, 26, 2, 1, 7, 18, 31, 32, 32, 45, 32],
             'conv': [12, 21, 40, 42, 43, 47, 49, 51, 60, 64, 64, 64], 'mhsa': [6, 11, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]},

        "double_prune_total_num_components_216_adjust_dropout_True_beta_0.1_gradient_update_step_1000":
            {'ff1': [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4], 'ff2': [0, 4, 3, 1, 2, 3, 4, 4, 4, 4, 4, 0],
             'conv': [2, 2, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8], 'mhsa': [8, 12, 11, 11, 8, 6, 9, 8, 7, 6, 6, 0]},
        "double_prune_total_num_components_216_noise_var_0_beta_0.1_gradient_update_step_1000":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [0, 4, 4, 0, 0, 0, 0, 4, 4, 4, 5, 4],
             'conv': [2, 2, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [8, 12, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6]},
        "double_prune_total_num_components_216_noise_var_0.1_beta_0.1_gradient_update_step_1000":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 8], 'ff2': [0, 4, 3, 3, 0, 0, 0, 2, 4, 4, 4, 4],
             'conv': [2, 2, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [11, 12, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6]},
        "double_prune_total_num_components_216_noise_var_0.0001_beta_0.1_gradient_update_step_1000":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 5], 'ff2': [2, 4, 2, 2, 0, 0, 1, 2, 4, 4, 4, 4],
             'conv': [2, 3, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [11, 12, 10, 7, 6, 6, 6, 6, 6, 6, 6, 4]},

        "double_prune_total_num_components_1224_adjust_dropout_True_beta_0.1_gradient_update_step_1000":
            {'ff1': [31, 35, 32, 32, 32, 31, 29, 32, 32, 32, 62, 32],
             'ff2': [18, 32, 26, 6, 2, 0, 11, 25, 32, 33, 32, 32],
             'conv': [16, 24, 47, 46, 47, 49, 49, 55, 60, 64, 64, 64], 'mhsa': [10, 12, 11, 8, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_1224_noise_var_0.1_beta_0.1_gradient_update_step_1000":
            {'ff1': [31, 35, 32, 32, 32, 31, 29, 32, 32, 32, 62, 32],
             'ff2': [18, 32, 26, 6, 2, 0, 11, 25, 32, 33, 32, 32],
             'conv': [16, 24, 47, 46, 47, 49, 49, 55, 60, 64, 64, 64], 'mhsa': [10, 12, 11, 8, 6, 6, 6, 6, 6, 6, 6, 0]},
        "double_prune_total_num_components_1224_noise_var_0.0001_beta_0.1_gradient_update_step_1000":
            {'ff1': [31, 35, 32, 32, 32, 31, 29, 32, 32, 32, 62, 32],
             'ff2': [18, 32, 26, 6, 2, 0, 11, 25, 32, 33, 32, 32],
             'conv': [16, 24, 47, 46, 47, 49, 49, 55, 60, 64, 64, 64], 'mhsa': [10, 12, 11, 8, 6, 6, 6, 6, 6, 6, 6, 0]}
    }

    for final_adapt_epochs in [90]:
        for pct in [0.15]:
            for num_iters in [1]:
                for adjust_dropout in [True]:
                    if pct / num_iters > 0.007:
                        lst_replace_pct = [pct / num_iters] * num_iters
                        num_epochs_per_iter = final_adapt_epochs // num_iters
                        adaptation_global_epochs = list(
                            range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
                        if len(adaptation_global_epochs) > len(lst_replace_pct):
                            adaptation_global_epochs = adaptation_global_epochs[:num_iters]
                        assert len(adaptation_global_epochs) == len(lst_replace_pct)

                        for grad_update_beta in [0.1]:
                            for grad_score_metric in ["first_taylor", "gradient"]:
                                for grad_score_update_steps in [1000]:
                                    for num_chunks in [4, 32]:
                                        if grad_score_metric == "gradient":
                                            num_chunks = 4
                                        for new_weights_noise_var in [0, 0.1, 0.0001]:
                                            total_num_components = (3 * num_chunks + 6) * 12
                                            network_args = {"specaug_args": {"time_min_num_masks": 2,
                                                                             "time_max_mask_per_n_frames": 25,
                                                                             "time_mask_max_size": 20,
                                                                             "freq_min_num_masks": 2,
                                                                             "freq_mask_max_size": 5,
                                                                             "freq_max_num_masks": 8},
                                                            "final_dropout": 0.1,
                                                            "att_weights_dropout": 0.1,
                                                            "dropout": 0.1,
                                                            "adjust_dropout": True,
                                                            "model_dim": 384,
                                                            "total_num_components": total_num_components,
                                                            "lst_cmp_cost": ([295584 * 4 // num_chunks] * num_chunks + [
                                                                114336 * 4 // num_chunks] * num_chunks + [98688] * 6 + [
                                                                                 295584 * 4 // num_chunks] * num_chunks) * 12}
                                            grad_score_opts = {
                                                "grad_score_update_steps": grad_score_update_steps,
                                                "grad_score_metric": grad_score_metric,
                                                "grad_update_beta": grad_update_beta
                                            }
                                            network_args["grad_score_opts"] = grad_score_opts
                                            adaptation_opts = {
                                                "adaptation_global_step": [1220 * i for i in adaptation_global_epochs],
                                                "total_cost": 42083471,
                                                "rest_cost": 1113743,
                                                "lst_replace_pct": lst_replace_pct,
                                                "dict_module_cost": {
                                                    "ff1": 295584 * 4 // num_chunks,
                                                    "conv": 114336 * 4 // num_chunks,
                                                    "mhsa": 98688,
                                                    "ff2": 295584 * 4 // num_chunks,
                                                },
                                                "new_weights_noise_var": new_weights_noise_var
                                            }
                                            network_args["adaptation_opts"] = adaptation_opts
                                            component_dist = {
                                                "ff1": [num_chunks] * 12,
                                                "conv": [num_chunks] * 12,
                                                "mhsa": [6] * 12,
                                                "ff2": [num_chunks] * 12,
                                            }
                                            network_args["component_dist"] = component_dist

                                            config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
                                                                           network_args=network_args)
                                            num_layers_12_experiment_name = f"double_prune_total_num_components_{total_num_components}_noise_var_{new_weights_noise_var}_beta_{grad_update_beta}_{grad_score_metric}_update_step_{grad_score_update_steps}"
                                            recog_network_args = copy.deepcopy(network_args)
                                            if num_layers_12_experiment_name in dict_recog_component_dist:
                                                recog_component_dist = dict_recog_component_dist[
                                                    num_layers_12_experiment_name]
                                            else:
                                                recog_component_dist = component_dist
                                            lst_ff1_h_dims = [i * ((384 * 4) // num_chunks) for i in
                                                              recog_component_dist["ff1"]]
                                            lst_ff2_h_dims = [i * ((384 * 4) // num_chunks) for i in
                                                              recog_component_dist["ff2"]]
                                            lst_channels = [i * ((96 * 4) // num_chunks) for i in
                                                            recog_component_dist["conv"]]
                                            lst_att_heads = recog_component_dist["mhsa"]
                                            recog_network_args["lst_ff1_h_dims"] = lst_ff1_h_dims
                                            recog_network_args["lst_ff2_h_dims"] = lst_ff2_h_dims
                                            recog_network_args["lst_att_heads"] = lst_att_heads
                                            recog_network_args["lst_channels"] = lst_channels
                                            total_num_components = 0
                                            for k, v in recog_component_dist.items():
                                                total_num_components += sum(v)
                                            recog_network_args["total_num_components"] = total_num_components
                                            recog_network_args["lst_cmp_cost"] = [0] * total_num_components

                                            recog_config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
                                                                                 network_args=recog_network_args)
                                            experiments[num_layers_12_experiment_name] = get_returnn_configs(config,
                                                                                                             recog_config)

    return experiments
