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
    from i6_experiments.users.jxu.experiments.ctc.swb.pytorch_networks.dynamic_adaptable_conformer.adapt_based_on_learnable_importance_scores_no_constraint import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(num_subepochs:int, train_data_config: dict, dev_data_config: dict, peak_lr: float) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.swb.pytorch_networks.dynamic_adaptable_conformer.adapt_based_on_learnable_importance_scores_no_constraint import get_default_config_v1

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
        "double_prune_total_num_components_216_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [4, 1, 0, 1, 2, 3, 2, 3, 4, 5, 5, 6], 'ff2': [8, 4, 4, 4, 4, 4, 4, 5, 8, 8, 8, 4],
             'conv': [4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4], 'mhsa': [6, 5, 4, 6, 5, 3, 5, 4, 4, 4, 5, 6]},
        "double_prune_total_num_components_216_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [4, 4, 0, 4, 4, 4, 0, 4, 4, 5, 7, 6], 'ff2': [6, 4, 4, 4, 4, 4, 4, 4, 4, 6, 12, 4],
             'conv': [4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'mhsa': [4, 6, 5, 6, 5, 4, 5, 4, 4, 1, 4, 1]},
        "double_prune_total_num_components_216_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 8], 'ff2': [4, 4, 4, 4, 4, 4, 4, 4, 0, 6, 9, 4],
             'conv': [4, 4, 4, 5, 5, 5, 5, 5, 5, 3, 4, 3], 'mhsa': [4, 6, 6, 6, 6, 6, 6, 5, 6, 3, 4, 4]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [31, 10, 16, 11, 6, 4, 8, 32, 34, 40, 46, 47],
             'ff2': [41, 32, 35, 34, 29, 29, 36, 43, 46, 64, 64, 41],
             'conv': [34, 30, 32, 33, 35, 32, 33, 32, 33, 34, 33, 36], 'mhsa': [6, 5, 6, 6, 6, 6, 4, 5, 6, 5, 5, 6]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [32, 24, 29, 20, 8, 4, 6, 32, 32, 34, 47, 40],
             'ff2': [25, 32, 32, 32, 25, 23, 32, 34, 34, 64, 113, 56],
             'conv': [32, 32, 32, 32, 32, 32, 32, 33, 34, 34, 32, 32], 'mhsa': [5, 6, 6, 4, 5, 5, 4, 5, 6, 4, 5, 0]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [32, 33, 32, 32, 32, 32, 32, 25, 42, 1, 51, 47],
             'ff2': [32, 32, 32, 32, 32, 33, 25, 1, 39, 70, 40, 38],
             'conv': [32, 36, 35, 36, 36, 36, 35, 34, 22, 33, 30, 28], 'mhsa': [3, 6, 6, 6, 5, 6, 6, 4, 6, 6, 4, 1]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.1_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [27, 12, 14, 10, 17, 16, 14, 29, 36, 45, 46, 38],
             'ff2': [45, 34, 41, 36, 35, 34, 37, 38, 43, 57, 57, 36],
             'conv': [34, 30, 27, 31, 33, 31, 31, 32, 35, 35, 32, 34], 'mhsa': [6, 4, 6, 6, 6, 6, 5, 3, 5, 3, 5, 6]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.1_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [31, 18, 27, 28, 18, 12, 6, 34, 32, 35, 54, 37],
             'ff2': [36, 32, 34, 33, 22, 15, 24, 39, 34, 63, 99, 43],
             'conv': [34, 33, 31, 32, 33, 32, 32, 33, 33, 34, 32, 32], 'mhsa': [6, 5, 6, 5, 5, 5, 4, 5, 6, 4, 5, 1]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.1_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [32, 30, 32, 33, 32, 32, 25, 0, 24, 37, 48, 42],
             'ff2': [32, 31, 29, 23, 32, 33, 1, 34, 44, 58, 70, 38],
             'conv': [29, 32, 34, 37, 34, 34, 34, 33, 34, 34, 32, 34], 'mhsa': [5, 6, 6, 6, 5, 6, 4, 6, 5, 4, 3, 4]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.2_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [32, 11, 18, 13, 9, 7, 10, 28, 38, 39, 43, 41],
             'ff2': [36, 35, 43, 35, 28, 26, 41, 40, 42, 57, 59, 41],
             'conv': [34, 36, 35, 33, 32, 33, 33, 33, 37, 35, 32, 34], 'mhsa': [6, 4, 6, 6, 6, 6, 5, 5, 6, 5, 6, 6]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.2_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [33, 15, 28, 32, 11, 7, 13, 31, 32, 34, 50, 39],
             'ff2': [38, 33, 32, 31, 23, 12, 34, 31, 37, 61, 96, 54],
             'conv': [33, 30, 31, 30, 32, 31, 31, 33, 33, 32, 33, 32], 'mhsa': [6, 6, 6, 6, 6, 5, 5, 4, 5, 4, 5, 0]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.2_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [31, 33, 32, 32, 33, 32, 32, 32, 11, 52, 35, 49],
             'ff2': [32, 32, 31, 32, 32, 32, 28, 24, 21, 52, 49, 40],
             'conv': [26, 21, 28, 35, 30, 37, 33, 34, 37, 36, 28, 33], 'mhsa': [3, 6, 6, 6, 6, 5, 3, 6, 4, 4, 6, 1]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.3_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [32, 19, 7, 13, 12, 7, 5, 32, 35, 39, 43, 41],
             'ff2': [51, 36, 34, 41, 32, 25, 36, 42, 45, 59, 60, 36],
             'conv': [33, 33, 30, 32, 36, 31, 34, 32, 37, 33, 32, 34], 'mhsa': [6, 5, 6, 6, 5, 6, 5, 5, 6, 4, 5, 6]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.3_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [32, 27, 27, 27, 1, 7, 7, 32, 32, 34, 53, 41],
             'ff2': [27, 32, 32, 33, 17, 23, 32, 33, 56, 65, 90, 50],
             'conv': [32, 32, 32, 32, 32, 32, 31, 32, 32, 32, 32, 32], 'mhsa': [6, 6, 6, 5, 5, 6, 5, 4, 6, 4, 3, 0]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.3_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [32, 33, 32, 32, 32, 34, 33, 35, 12, 3, 21, 47],
             'ff2': [32, 32, 30, 28, 26, 30, 42, 42, 37, 40, 61, 48],
             'conv': [33, 33, 28, 36, 34, 37, 38, 35, 35, 37, 36, 34], 'mhsa': [3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 1]}
    }

    for final_adapt_epochs in [60,]:
    # for final_adapt_epochs in [50]:
        for pct in [0.15]:
        # for pct in [0.15]:
            for num_iters in [1, 4, 16]:
            # for num_iters in [1]:
                for adjust_dropout in [True]:
                    if pct / num_iters > 0.0072:
                        lst_replace_pct = [pct / num_iters] * num_iters
                        num_epochs_per_iter = final_adapt_epochs // num_iters
                        adaptation_global_epochs = list(
                            range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
                        if len(adaptation_global_epochs) > len(lst_replace_pct):
                            adaptation_global_epochs = adaptation_global_epochs[:num_iters]
                        assert len(adaptation_global_epochs) == len(lst_replace_pct)

                        for num_chunks in [4, 32]:
                            for learn_score_random_weight in [0.5]:
                                total_num_components = (3 * num_chunks + 6) * 12
                                network_args = {"specaug_args": {"time_min_num_masks": 2,
                                                                 "time_max_mask_per_n_frames": 25,
                                                                 "time_mask_max_size": 20,
                                                                 "freq_min_num_masks": 2,
                                                                 "freq_mask_max_size": 5,
                                                                 "freq_max_num_masks": 16},
                                                "final_dropout": 0.2,
                                                "att_weights_dropout": 0.1,
                                                "dropout": 0.1,
                                                "adjust_dropout": adjust_dropout,
                                                "model_dim": 384,
                                                "total_num_components": total_num_components,
                                                "lst_cmp_cost": ([295584*4//num_chunks] * num_chunks + [114336*4//num_chunks] * num_chunks + [98688] * 6 + [
                                                    295584*4//num_chunks] * num_chunks) * 12}

                                adaptation_opts = {
                                    "adaptation_global_step": [1200 * i for i in adaptation_global_epochs],
                                    "total_cost": 42083471,
                                    "rest_cost": 1113743,
                                    "lst_replace_pct": lst_replace_pct,
                                    "dict_module_cost": {
                                        "ff1": 295584 * 4 // num_chunks,
                                        "conv": 114336 * 4 // num_chunks,
                                        "mhsa": 98688,
                                        "ff2": 295584 * 4 // num_chunks,
                                    },
                                    "learn_score_random_weight": learn_score_random_weight
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
                                str_adapt_epochs = "_".join([str(i) for i in adaptation_global_epochs])
                                repeat_times = lst_replace_pct.count(lst_replace_pct[0])
                                str_replace_pct = f"{str(lst_replace_pct[0])}_{repeat_times}"
                                # str_replace_pct = "_".join([str(i) for i in lst_replace_pct])
                                num_layers_12_experiment_name = f"double_prune_total_num_components_{total_num_components}_learn_score_random_weight_{learn_score_random_weight}_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
                                recog_network_args = copy.deepcopy(network_args)
                                if num_layers_12_experiment_name in dict_recog_component_dist:
                                    recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
                                else:
                                    recog_component_dist = component_dist
                                lst_ff1_h_dims = [i * ((384 * 4) // num_chunks) for i in recog_component_dist["ff1"]]
                                lst_ff2_h_dims = [i * ((384 * 4) // num_chunks) for i in recog_component_dist["ff2"]]
                                lst_channels = [i * ((96 * 4) // num_chunks) for i in recog_component_dist["conv"]]
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

    for final_adapt_epochs in [60, ]:
        # for final_adapt_epochs in [50]:
        for pct in [0.15]:
            # for pct in [0.15]:
            for num_iters in [1, 4, 16]:
                # for num_iters in [1]:
                for adjust_dropout in [True]:
                    if pct / num_iters > 0.0072:
                        lst_replace_pct = [pct / num_iters] * num_iters
                        num_epochs_per_iter = final_adapt_epochs // num_iters
                        adaptation_global_epochs = list(
                            range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
                        if len(adaptation_global_epochs) > len(lst_replace_pct):
                            adaptation_global_epochs = adaptation_global_epochs[:num_iters]
                        assert len(adaptation_global_epochs) == len(lst_replace_pct)

                        for num_chunks in [32]:
                            for learn_score_random_weight in [0.3, 0.2, 0.1]:
                                total_num_components = (3 * num_chunks + 6) * 12
                                network_args = {"specaug_args": {"time_min_num_masks": 2,
                                                                 "time_max_mask_per_n_frames": 25,
                                                                 "time_mask_max_size": 20,
                                                                 "freq_min_num_masks": 2,
                                                                 "freq_mask_max_size": 5,
                                                                 "freq_max_num_masks": 16},
                                                "final_dropout": 0.2,
                                                "att_weights_dropout": 0.1,
                                                "dropout": 0.1,
                                                "adjust_dropout": adjust_dropout,
                                                "model_dim": 384,
                                                "total_num_components": total_num_components,
                                                "lst_cmp_cost": ([295584 * 4 // num_chunks] * num_chunks + [
                                                    114336 * 4 // num_chunks] * num_chunks + [98688] * 6 + [
                                                                     295584 * 4 // num_chunks] * num_chunks) * 12}

                                adaptation_opts = {
                                    "adaptation_global_step": [1200 * i for i in adaptation_global_epochs],
                                    "total_cost": 42083471,
                                    "rest_cost": 1113743,
                                    "lst_replace_pct": lst_replace_pct,
                                    "dict_module_cost": {
                                        "ff1": 295584 * 4 // num_chunks,
                                        "conv": 114336 * 4 // num_chunks,
                                        "mhsa": 98688,
                                        "ff2": 295584 * 4 // num_chunks,
                                    },
                                    "learn_score_random_weight": learn_score_random_weight
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
                                str_adapt_epochs = "_".join([str(i) for i in adaptation_global_epochs])
                                repeat_times = lst_replace_pct.count(lst_replace_pct[0])
                                str_replace_pct = f"{str(lst_replace_pct[0])}_{repeat_times}"
                                # str_replace_pct = "_".join([str(i) for i in lst_replace_pct])
                                num_layers_12_experiment_name = f"double_prune_total_num_components_{total_num_components}_learn_score_random_weight_{learn_score_random_weight}_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
                                recog_network_args = copy.deepcopy(network_args)
                                if num_layers_12_experiment_name in dict_recog_component_dist:
                                    recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
                                else:
                                    recog_component_dist = component_dist
                                lst_ff1_h_dims = [i * ((384 * 4) // num_chunks) for i in recog_component_dist["ff1"]]
                                lst_ff2_h_dims = [i * ((384 * 4) // num_chunks) for i in recog_component_dist["ff2"]]
                                lst_channels = [i * ((96 * 4) // num_chunks) for i in recog_component_dist["conv"]]
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