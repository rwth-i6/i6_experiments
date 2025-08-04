import copy
from typing import Any, Dict, List, Optional, Union
import functools

import i6_core.returnn as returnn
import i6_core.rasr as rasr
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection

from i6_experiments.common.tools.sctk import compile_sctk

from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, ConfigVariant
import i6_experiments.users.jxu.experiments.ctc.tedlium2.configs.configs_helper as configs_helper


from sisyphus import gs, tk


# ********** Settings **********
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 88
num_subepochs = 300

# ********** Return Config generators **********

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
    from i6_experiments.users.jxu.experiments.ctc.swb.pytorch_networks.dynamic_adaptable_conformer.adapt_based_on_gradient_ranking_finer_granul_double_and_prune import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(num_subepochs:int, train_data_config: dict, dev_data_config: dict, peak_lr: float) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.swb.pytorch_networks.dynamic_adaptable_conformer.adapt_based_on_gradient_ranking_finer_granul_double_and_prune import get_default_config_v1

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
        cycle_epoch=(num_subepochs-30)//2,
        lr_1= peak_lr / 100,
        lr_2 = peak_lr / 10,
        peak_lr=peak_lr,
        final_lr=1e-08,
        batch_size=18000 * 80,
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
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.15_iters_1_0.0003":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [0, 2, 2, 0, 4, 4, 4, 4, 4, 4, 8, 4],
             'conv': [0, 0, 0, 2, 3, 5, 6, 6, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 11, 8]},
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.15_iters_4_0.0003":
            {'ff1': [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7], 'ff2': [0, 2, 4, 4, 4, 4, 1, 4, 1, 4, 4, 4],
             'conv': [0, 0, 1, 4, 3, 6, 6, 7, 7, 9, 16, 13], 'mhsa': [10, 8, 8, 8, 8, 9, 8, 8, 8, 10, 8, 8]},
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.15_iters_8_0.0003":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [0, 4, 4, 4, 1, 4, 4, 4, 0, 2, 4, 4],
             'conv': [0, 0, 1, 1, 3, 4, 6, 6, 7, 9, 15, 12], 'mhsa': [16, 8, 12, 8, 10, 8, 8, 8, 8, 8, 8, 16]},

        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.2_iters_1_0.0003":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 7, 8, 8], 'ff2': [0, 2, 0, 0, 0, 4, 4, 4, 4, 4, 8, 6],
             'conv': [0, 0, 0, 1, 2, 4, 6, 6, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 8, 8, 8, 8, 10, 14, 16, 8]},
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.2_iters_4_0.0003":
            {'ff1': [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4], 'ff2': [0, 0, 0, 4, 4, 4, 4, 4, 0, 1, 4, 4],
             'conv': [0, 0, 0, 2, 4, 6, 6, 7, 8, 9, 16, 14], 'mhsa': [16, 13, 8, 8, 11, 8, 12, 9, 12, 8, 12, 8]},
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.2_iters_8_0.0003":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [0, 0, 4, 4, 0, 4, 4, 3, 1, 3, 4, 4],
             'conv': [0, 0, 1, 4, 4, 5, 7, 9, 7, 10, 14, 15], 'mhsa': [11, 8, 18, 12, 9, 11, 10, 8, 8, 8, 8, 16]},

        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.25_iters_1_0.0003":
            {'ff1': [0, 4, 4, 2, 4, 4, 4, 4, 4, 8, 8, 8], 'ff2': [0, 0, 0, 0, 0, 3, 4, 4, 4, 5, 8, 8],
             'conv': [0, 0, 0, 0, 2, 2, 6, 6, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 8, 8, 15, 10, 8, 15, 16, 8]},
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.25_iters_4_0.0003":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 6], 'ff2': [0, 0, 4, 0, 4, 4, 1, 2, 0, 4, 4, 4],
             'conv': [0, 0, 1, 1, 2, 5, 6, 7, 9, 13, 17, 14], 'mhsa': [10, 16, 16, 8, 9, 8, 11, 8, 16, 12, 8, 8]},
        "double_prune_total_num_components_240_adapt_epochs_60_pct_0.25_iters_8_0.0003":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4], 'ff2': [0, 0, 4, 0, 0, 3, 4, 4, 0, 3, 4, 4],
             'conv': [0, 0, 1, 2, 4, 3, 6, 8, 9, 7, 11, 14], 'mhsa': [18, 14, 16, 8, 12, 15, 8, 8, 8, 8, 8, 16]},

        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.15_iters_1_0.0003":
            {'ff1': [0, 32, 32, 32, 32, 32, 32, 32, 32, 48, 64, 64],
             'ff2': [0, 21, 14, 5, 31, 32, 32, 32, 32, 35, 51, 33],
             'conv': [0, 0, 8, 16, 17, 32, 47, 52, 61, 64, 64, 64], 'mhsa': [8, 8, 8, 8, 8, 8, 8, 8, 9, 8, 8, 8]},
        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.15_iters_4_0.0003":
            {'ff1': [5, 32, 32, 32, 32, 32, 32, 32, 32, 32, 41, 43],
             'ff2': [0, 32, 32, 29, 32, 32, 23, 20, 16, 27, 32, 32],
             'conv': [1, 0, 4, 11, 34, 35, 51, 53, 58, 80, 94, 103], 'mhsa': [16, 11, 8, 8, 8, 8, 8, 8, 8, 8, 12, 8]},
        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.15_iters_8_0.0003":
            {'ff1': [22, 32, 32, 32, 32, 32, 32, 32, 32, 32, 35, 36],
             'ff2': [0, 32, 24, 16, 11, 32, 21, 27, 27, 32, 32, 32],
             'conv': [0, 1, 2, 12, 34, 37, 48, 52, 72, 93, 120, 111], 'mhsa': [13, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16]},

        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.2_iters_1_0.0003":
            {'ff1': [0, 32, 32, 32, 32, 32, 32, 32, 32, 42, 64, 64],
             'ff2': [0, 16, 14, 0, 0, 30, 32, 32, 32, 38, 64, 51], 'conv': [0, 0, 1, 2, 10, 28, 44, 52, 63, 64, 64, 64],
             'mhsa': [8, 8, 8, 8, 8, 8, 8, 8, 8, 15, 16, 8]},
        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.2_iters_4_0.0003":
            {'ff1': [0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 49, 42],
             'ff2': [0, 32, 32, 9, 26, 32, 14, 13, 13, 32, 32, 32],
             'conv': [0, 2, 10, 19, 32, 41, 49, 52, 65, 91, 113, 94], 'mhsa': [16, 15, 16, 9, 8, 8, 8, 8, 10, 8, 8, 8]},
        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.2_iters_8_0.0003":
            {'ff1': [28, 32, 32, 32, 32, 32, 32, 32, 32, 32, 34, 34],
             'ff2': [0, 32, 31, 0, 7, 29, 16, 27, 25, 32, 32, 32],
             'conv': [2, 1, 4, 32, 35, 26, 41, 60, 72, 86, 114, 117], 'mhsa': [14, 9, 8, 8, 8, 8, 10, 12, 8, 8, 8, 16]},

        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.25_iters_1_0.0003":
            {'ff1': [0, 32, 28, 30, 22, 32, 32, 32, 45, 64, 64, 64], 'ff2': [0, 1, 2, 0, 1, 16, 32, 32, 32, 63, 64, 36],
             'conv': [0, 0, 1, 15, 16, 35, 48, 61, 64, 64, 64, 64], 'mhsa': [4, 8, 8, 8, 8, 8, 8, 8, 16, 9, 16, 8]},
        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.25_iters_4_0.0003":
            {'ff1': [12, 32, 31, 31, 31, 32, 32, 32, 32, 32, 58, 56],
             'ff2': [0, 3, 1, 0, 12, 9, 27, 32, 32, 32, 32, 32],
             'conv': [1, 0, 13, 13, 16, 26, 42, 63, 70, 101, 120, 121],
             'mhsa': [16, 16, 16, 13, 9, 8, 8, 8, 10, 10, 8, 8]},
        "double_prune_total_num_components_1248_adapt_epochs_60_pct_0.25_iters_8_0.0003":
            {'ff1': [18, 32, 32, 32, 32, 32, 32, 32, 32, 34, 49, 34],
             'ff2': [0, 32, 0, 0, 1, 29, 24, 20, 17, 25, 41, 32],
             'conv': [1, 3, 5, 11, 8, 43, 46, 64, 75, 99, 124, 126], 'mhsa': [17, 13, 15, 8, 13, 8, 9, 8, 8, 8, 10, 16]}

    }

    for final_adapt_epochs in [60,]:
        for pct in [0.15, 0.2, 0.25]:
            for num_iters in [1, 4, 8]:
                for adjust_dropout in [True]:
                    for dropout in [0.1]:
                        if pct / num_iters > 0.0072:
                            lst_replace_pct = [pct / num_iters] * num_iters
                            num_epochs_per_iter = final_adapt_epochs // num_iters
                            adaptation_global_epochs = list(
                                range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
                            if len(adaptation_global_epochs) > len(lst_replace_pct):
                                adaptation_global_epochs = adaptation_global_epochs[:num_iters]
                            assert len(adaptation_global_epochs) == len(lst_replace_pct)

                            for grad_update_beta in [0.1]:
                                for grad_score_metric in ["first_taylor"]:
                                    network_args = {"specaug_args": {"time_min_num_masks": 2,
                                                                     "time_max_mask_per_n_frames": 25,
                                                                     "time_mask_max_size": 20,
                                                                     "freq_min_num_masks": 2,
                                                                     "freq_mask_max_size": 5,
                                                                     "freq_max_num_masks": 16},
                                                    "final_dropout": 0.2,
                                                    "att_weights_dropout": dropout,
                                                    "dropout": dropout,
                                                    "adjust_dropout": adjust_dropout,
                                                    "model_dim": 512,
                                                    "total_num_components": 240,
                                                    "lst_cmp_cost": ([525184] * 4 + [201600] * 4 + [131456] * 8 + [525184] * 4) * 12}

                                    grad_score_opts = {
                                        "grad_score_update_steps": 900,
                                        "grad_score_metric": grad_score_metric,
                                        "grad_update_beta": grad_update_beta
                                    }
                                    network_args["grad_score_opts"] = grad_score_opts
                                    adaptation_opts = {
                                        "adaptation_global_step": [1200 * i for i in adaptation_global_epochs],
                                        "total_cost": 74165392,
                                        "rest_cost": 1451152,
                                        "lst_replace_pct": lst_replace_pct,
                                        "dict_module_cost": {
                                            "ff1": 525184,
                                            "conv": 201600,
                                            "mhsa": 131456,
                                            "ff2": 525184,
                                        }
                                    }
                                    network_args["adaptation_opts"] = adaptation_opts
                                    component_dist = {
                                        "ff1": [4] * 12,
                                        "conv": [4] * 12,
                                        "mhsa": [8] * 12,
                                        "ff2": [4] * 12,
                                    }
                                    network_args["component_dist"] = component_dist

                                    config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
                                                                   network_args=network_args)
                                    str_adapt_epochs = "_".join([str(i) for i in adaptation_global_epochs])
                                    repeat_times = lst_replace_pct.count(lst_replace_pct[0])
                                    str_replace_pct = f"{str(lst_replace_pct[0])}_{repeat_times}"
                                    # str_replace_pct = "_".join([str(i) for i in lst_replace_pct])
                                    num_layers_12_experiment_name = f"double_prune_total_num_components_240_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
                                    recog_network_args = copy.deepcopy(network_args)
                                    if num_layers_12_experiment_name in dict_recog_component_dist:
                                        recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
                                    else:
                                        recog_component_dist = component_dist
                                    lst_ff1_h_dims = [i * 512 for i in recog_component_dist["ff1"]]
                                    lst_ff2_h_dims = [i * 512 for i in recog_component_dist["ff2"]]
                                    lst_att_heads = recog_component_dist["mhsa"]
                                    lst_channels = [i * 128 for i in recog_component_dist["conv"]]
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
                                    experiments[num_layers_12_experiment_name] = get_returnn_configs(config, recog_config)


    for final_adapt_epochs in [60,]:
        for pct in [0.15, 0.2, 0.25]:
            for num_iters in [1, 4, 8]:
                for adjust_dropout in [True]:
                    for dropout in [0.1]:
                        if pct / num_iters > 0.0072:
                            lst_replace_pct = [pct / num_iters] * num_iters
                            num_epochs_per_iter = final_adapt_epochs // num_iters
                            adaptation_global_epochs = list(
                                range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
                            if len(adaptation_global_epochs) > len(lst_replace_pct):
                                adaptation_global_epochs = adaptation_global_epochs[:num_iters]
                            assert len(adaptation_global_epochs) == len(lst_replace_pct)

                            for grad_update_beta in [0.1]:
                                for grad_score_metric in ["first_taylor"]:
                                    for num_chunks in [32]:
                                        total_num_components = (3 * num_chunks + 8) * 12
                                        network_args = {"specaug_args": {"time_min_num_masks": 2,
                                                                         "time_max_mask_per_n_frames": 25,
                                                                         "time_mask_max_size": 20,
                                                                         "freq_min_num_masks": 2,
                                                                         "freq_mask_max_size": 5,
                                                                         "freq_max_num_masks": 16},
                                                        "final_dropout": 0.2,
                                                        "att_weights_dropout": dropout,
                                                        "dropout": dropout,
                                                        "adjust_dropout": adjust_dropout,
                                                        "model_dim": 512,
                                                        "total_num_components": total_num_components,
                                                        "lst_cmp_cost": ([525184*4//num_chunks]*num_chunks + [201600*4//num_chunks] * num_chunks + [131456] * 8 + [525184*4//num_chunks]*num_chunks) * 12}

                                        grad_score_opts = {
                                            "grad_score_update_steps": 1000,
                                            "grad_score_metric": grad_score_metric,
                                            "grad_update_beta": grad_update_beta
                                        }
                                        network_args["grad_score_opts"] = grad_score_opts
                                        adaptation_opts = {
                                            "adaptation_global_step": [1200 * i for i in adaptation_global_epochs],
                                            "total_cost": 74165392,
                                            "rest_cost": 1451152,
                                            "lst_replace_pct": lst_replace_pct,
                                            "dict_module_cost": {
                                                "ff1": 525184*4//num_chunks,
                                                "conv": 201600*4//num_chunks,
                                                "mhsa": 131456,
                                                "ff2": 525184*4//num_chunks,
                                            }
                                        }
                                        network_args["adaptation_opts"] = adaptation_opts
                                        component_dist = {
                                            "ff1": [num_chunks] * 12,
                                            "conv": [num_chunks] * 12,
                                            "mhsa": [8] * 12,
                                            "ff2": [num_chunks] * 12,
                                        }
                                        network_args["component_dist"] = component_dist

                                        config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
                                                                       network_args=network_args)
                                        str_adapt_epochs = "_".join([str(i) for i in adaptation_global_epochs])
                                        repeat_times = lst_replace_pct.count(lst_replace_pct[0])
                                        str_replace_pct = f"{str(lst_replace_pct[0])}_{repeat_times}"
                                        # str_replace_pct = "_".join([str(i) for i in lst_replace_pct])
                                        num_layers_12_experiment_name = f"double_prune_total_num_components_{total_num_components}_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
                                        recog_network_args = copy.deepcopy(network_args)
                                        if num_layers_12_experiment_name in dict_recog_component_dist:
                                            recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
                                        else:
                                            recog_component_dist = component_dist
                                        lst_ff1_h_dims = [i * ((512*4)//num_chunks) for i in recog_component_dist["ff1"]]
                                        lst_ff2_h_dims = [i * ((512*4)//num_chunks) for i in recog_component_dist["ff2"]]
                                        lst_channels = [i * ((128*4)//num_chunks) for i in recog_component_dist["conv"]]
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
                                        experiments[num_layers_12_experiment_name] = get_returnn_configs(config, recog_config)

    return experiments