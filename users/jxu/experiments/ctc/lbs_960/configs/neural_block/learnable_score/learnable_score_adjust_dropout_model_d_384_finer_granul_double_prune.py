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
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_conformer.adapt_based_on_learnable_importance_scores_no_constraint import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(train_data_config: dict, dev_data_config: dict, peak_lr: float) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_conformer.adapt_based_on_learnable_importance_scores_no_constraint import \
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
        "double_prune_total_num_components_216_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [4, 4, 3, 3, 3, 1, 4, 4, 4, 4, 7, 7], 'ff2': [4, 4, 3, 4, 3, 5, 5, 6, 8, 8, 8, 4],
             'conv': [3, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 'mhsa': [6, 4, 4, 5, 1, 2, 3, 4, 3, 1, 1, 1]},
        "double_prune_total_num_components_216_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [4, 3, 4, 4, 4, 4, 4, 4, 4, 6, 7, 12], 'ff2': [4, 4, 4, 4, 4, 4, 4, 4, 0, 6, 8, 4],
             'conv': [1, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'mhsa': [0, 6, 6, 5, 1, 4, 5, 5, 5, 1, 2, 4]},
        "double_prune_total_num_components_216_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6], 'ff2': [4, 4, 4, 4, 4, 4, 5, 5, 6, 9, 8, 6],
             'conv': [1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'mhsa': [2, 6, 5, 6, 6, 6, 6, 6, 6, 5, 4, 5]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [31, 32, 25, 22, 27, 25, 29, 21, 32, 40, 58, 57],
             'ff2': [32, 29, 29, 30, 29, 30, 29, 41, 61, 64, 63, 39],
             'conv': [26, 29, 29, 32, 30, 30, 31, 30, 32, 31, 32, 32], 'mhsa': [5, 5, 3, 2, 3, 2, 3, 3, 2, 3, 3, 1]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [32, 2, 32, 32, 32, 32, 32, 32, 37, 36, 48, 53],
             'ff2': [7, 32, 32, 32, 32, 32, 32, 39, 21, 58, 86, 50],
             'conv': [13, 24, 26, 29, 30, 28, 23, 30, 31, 31, 31, 32], 'mhsa': [0, 5, 6, 6, 4, 4, 6, 2, 3, 6, 4, 3]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.5_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [3, 32, 32, 32, 32, 32, 12, 22, 26, 38, 40, 43],
             'ff2': [32, 32, 32, 32, 32, 31, 38, 37, 39, 52, 58, 58],
             'conv': [12, 29, 32, 32, 32, 32, 32, 33, 33, 22, 31, 25], 'mhsa': [2, 6, 6, 6, 6, 6, 5, 6, 6, 5, 5, 5]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.3_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [32, 32, 23, 24, 22, 24, 23, 22, 37, 45, 48, 48],
             'ff2': [35, 29, 28, 29, 29, 30, 29, 46, 61, 63, 63, 38],
             'conv': [28, 27, 28, 32, 29, 29, 29, 31, 32, 31, 32, 33], 'mhsa': [6, 4, 4, 3, 4, 3, 2, 4, 2, 4, 4, 1]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.3_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [32, 0, 32, 32, 32, 32, 32, 32, 25, 42, 40, 61],
             'ff2': [32, 32, 32, 32, 32, 32, 32, 16, 51, 39, 85, 47],
             'conv': [16, 25, 20, 26, 31, 26, 25, 31, 32, 32, 30, 33], 'mhsa': [0, 6, 6, 5, 5, 5, 5, 3, 2, 5, 4, 3]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.3_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [32, 32, 32, 32, 32, 32, 5, 28, 24, 49, 38, 42],
             'ff2': [32, 32, 32, 33, 32, 28, 30, 40, 42, 46, 47, 62],
             'conv': [3, 26, 25, 32, 28, 27, 25, 31, 31, 22, 28, 24], 'mhsa': [2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 5]},

        "double_prune_total_num_components_1224_learn_score_random_weight_0.1_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [29, 29, 22, 27, 22, 23, 24, 23, 44, 44, 53, 43],
             'ff2': [33, 27, 30, 29, 31, 27, 26, 43, 55, 62, 62, 40],
             'conv': [26, 31, 27, 32, 28, 29, 28, 27, 31, 32, 33, 34], 'mhsa': [5, 4, 4, 5, 2, 2, 4, 5, 3, 6, 4, 2]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.1_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [14, 17, 32, 32, 31, 32, 32, 32, 35, 38, 54, 66],
             'ff2': [16, 32, 32, 5, 32, 32, 32, 37, 60, 49, 72, 46],
             'conv': [12, 21, 22, 24, 28, 26, 25, 23, 31, 30, 31, 31], 'mhsa': [0, 5, 5, 4, 6, 5, 4, 4, 3, 6, 4, 4]},
        "double_prune_total_num_components_1224_learn_score_random_weight_0.1_adapt_epochs_60_pct_0.15_iters_16_0.0004":
            {'ff1': [7, 34, 32, 33, 19, 26, 34, 26, 42, 39, 21, 48],
             'ff2': [33, 32, 32, 34, 19, 35, 35, 30, 43, 60, 49, 57],
             'conv': [11, 29, 32, 32, 24, 25, 31, 32, 30, 25, 33, 33], 'mhsa': [1, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 6]}
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
                                "learn_score_random_weight": 0.5
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
                            num_layers_12_experiment_name = f"double_prune_total_num_components_{total_num_components}_learn_score_random_weight_0.5_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
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

                        for num_chunks in [32]:
                            for learn_score_random_weight in [0.3, 0.1]:
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

    return experiments