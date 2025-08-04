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
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.15_iters_1_0.0004":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'ff2': [0, 4, 4, 4, 2, 2, 2, 4, 4, 5, 8, 4],
             'conv': [0, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8], 'mhsa': [8, 9, 8, 8, 8, 8, 7, 5, 0, 7, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.15_iters_2_0.0004":
            {'ff1': [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 8], 'ff2': [0, 4, 4, 4, 0, 0, 1, 4, 4, 4, 4, 4],
             'conv': [1, 4, 4, 5, 6, 5, 5, 6, 7, 8, 12, 14], 'mhsa': [16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.15_iters_4_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6], 'ff2': [1, 4, 4, 4, 0, 1, 0, 4, 0, 4, 8, 4],
             'conv': [1, 4, 4, 5, 4, 5, 6, 7, 9, 9, 11, 14], 'mhsa': [16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.15_iters_8_0.0004":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6], 'ff2': [0, 4, 4, 4, 0, 4, 4, 0, 0, 6, 5, 4],
             'conv': [1, 4, 4, 5, 6, 5, 6, 6, 7, 10, 12, 12], 'mhsa': [16, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.15_iters_16_0.0004":
            {'ff1': [4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5], 'ff2': [0, 4, 2, 4, 4, 4, 4, 0, 0, 6, 6, 2],
             'conv': [0, 4, 5, 1, 5, 6, 5, 6, 9, 10, 13, 15], 'mhsa': [15, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3]},

        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.2_iters_1_0.0004":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8], 'ff2': [0, 4, 4, 0, 0, 0, 3, 4, 4, 8, 7, 4],
             'conv': [0, 2, 4, 4, 5, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 7, 7, 8, 6, 3, 8, 7, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.2_iters_2_0.0004":
            {'ff1': [3, 8, 4, 4, 4, 4, 4, 4, 4, 4, 7, 8], 'ff2': [0, 4, 4, 4, 0, 2, 0, 1, 4, 4, 4, 4],
             'conv': [2, 3, 5, 5, 5, 5, 5, 6, 7, 8, 10, 11], 'mhsa': [16, 16, 11, 8, 8, 8, 8, 8, 1, 8, 0, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.2_iters_4_0.0004":
            {'ff1': [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 8], 'ff2': [0, 4, 4, 4, 4, 0, 0, 4, 0, 2, 7, 4],
             'conv': [0, 4, 4, 5, 5, 5, 5, 5, 6, 10, 13, 16], 'mhsa': [16, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.2_iters_8_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 7], 'ff2': [0, 4, 4, 0, 0, 0, 4, 0, 4, 4, 8, 4],
             'conv': [1, 4, 1, 5, 5, 5, 6, 7, 7, 9, 13, 13], 'mhsa': [16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.2_iters_16_0.0004":
            {'ff1': [1, 6, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6], 'ff2': [0, 4, 2, 0, 4, 4, 0, 0, 4, 6, 7, 3],
             'conv': [0, 4, 4, 6, 5, 6, 5, 7, 8, 12, 13, 15], 'mhsa': [16, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2]},

        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.25_iters_1_0.0004":
            {'ff1': [0, 8, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8], 'ff2': [0, 4, 1, 1, 0, 0, 3, 4, 4, 8, 8, 4],
             'conv': [0, 2, 3, 6, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [8, 15, 9, 8, 8, 0, 8, 1, 0, 5, 3, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.25_iters_2_0.0004":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 9], 'ff2': [0, 4, 4, 0, 0, 1, 0, 4, 4, 4, 8, 4],
             'conv': [1, 1, 4, 4, 4, 4, 5, 6, 8, 9, 14, 16], 'mhsa': [10, 16, 14, 8, 8, 8, 8, 3, 4, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.25_iters_4_0.0004":
            {'ff1': [1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 8], 'ff2': [0, 4, 4, 4, 0, 0, 0, 0, 0, 6, 8, 4],
             'conv': [0, 4, 4, 5, 5, 6, 6, 6, 6, 11, 14, 15], 'mhsa': [16, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.25_iters_8_0.0004":
            {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 5, 7, 8], 'ff2': [0, 4, 4, 0, 4, 4, 4, 0, 0, 0, 8, 4],
             'conv': [0, 4, 1, 6, 6, 6, 6, 5, 6, 10, 12, 13], 'mhsa': [17, 14, 16, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.25_iters_16_0.0004":
            {'ff1': [0, 5, 4, 4, 4, 4, 4, 4, 5, 5, 7, 7], 'ff2': [0, 4, 0, 4, 4, 4, 4, 0, 0, 7, 6, 4],
             'conv': [0, 4, 1, 6, 3, 7, 5, 7, 4, 10, 16, 12], 'mhsa': [15, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_120_pct_0.25_iters_32_0.0004":
            {'ff1': [5, 5, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7], 'ff2': [0, 2, 0, 4, 4, 4, 4, 0, 0, 5, 5, 5],
             'conv': [0, 6, 6, 1, 2, 5, 8, 7, 10, 12, 13, 13], 'mhsa': [18, 15, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9]},

        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_30_pct_0.15_iters_1_0.0004":
            {'ff1': [8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [6, 4, 4, 3, 0, 0, 0, 0, 0, 4, 4, 4],
             'conv': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6], 'mhsa': [16, 16, 14, 11, 8, 8, 8, 8, 8, 8, 8, 8]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_30_pct_0.15_iters_4_0.0004":
            {'ff1': [8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 2, 0, 4, 4, 4, 1, 0, 0, 0, 4, 4],
             'conv': [6, 6, 6, 8, 6, 7, 5, 9, 9, 9, 8, 6], 'mhsa': [22, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_30_pct_0.15_iters_6_0.0004":
            {'ff1': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 1, 0, 4, 4, 4, 4, 0, 0, 3, 4, 4],
             'conv': [2, 3, 6, 6, 7, 10, 3, 9, 8, 8, 9, 6], 'mhsa': [28, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]},

        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_60_pct_0.15_iters_1_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7], 'ff2': [4, 4, 2, 0, 0, 0, 0, 3, 4, 4, 4, 4],
             'conv': [2, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8], 'mhsa': [16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 4]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_60_pct_0.15_iters_4_0.0004":
            {'ff1': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 4, 0, 0, 0, 4, 0, 4, 4, 4, 4, 4],
             'conv': [1, 4, 5, 6, 7, 5, 4, 8, 9, 10, 12, 12], 'mhsa': [16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_60_pct_0.15_iters_6_0.0004":
            {'ff1': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 4, 0, 0, 2, 4, 4, 1, 0, 4, 4, 4],
             'conv': [1, 4, 3, 6, 4, 6, 6, 8, 7, 14, 15, 13], 'mhsa': [16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7]},

        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_90_pct_0.15_iters_1_0.0004":
            {'ff1': [4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 6, 8], 'ff2': [0, 4, 4, 4, 0, 0, 0, 2, 4, 4, 4, 4],
             'conv': [2, 4, 4, 6, 6, 6, 6, 6, 7, 8, 8, 8], 'mhsa': [14, 16, 13, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_90_pct_0.15_iters_4_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8], 'ff2': [4, 4, 4, 4, 0, 4, 0, 0, 0, 4, 4, 4],
             'conv': [1, 4, 5, 5, 4, 7, 5, 6, 7, 8, 13, 13], 'mhsa': [16, 13, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
        "double_prune_total_num_components_240_freq_mask_50_adapt_epochs_90_pct_0.15_iters_6_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7], 'ff2': [0, 4, 4, 4, 0, 4, 0, 2, 0, 4, 4, 4],
             'conv': [1, 4, 4, 5, 5, 6, 5, 6, 7, 10, 11, 16], 'mhsa': [16, 15, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]}
    }

    # for final_adapt_epochs in [120]:
    #     for pct in [0.15, 0.2, 0.25]:
    #         for num_iters in [1, 2, 4, 8, 16, 32]:
    #             for adjust_dropout in [True]:
    #                 if pct / num_iters > 0.007:
    #                     lst_replace_pct = [pct / num_iters] * num_iters
    #                     num_epochs_per_iter = final_adapt_epochs // num_iters
    #                     adaptation_global_epochs = list(
    #                         range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
    #                     if len(adaptation_global_epochs) > len(lst_replace_pct):
    #                         adaptation_global_epochs = adaptation_global_epochs[:num_iters]
    #                     assert len(adaptation_global_epochs) == len(lst_replace_pct)
    #
    #                     for grad_update_beta in [0.1]:
    #                         for grad_score_metric in ["first_taylor"]:
    #                             network_args = {"specaug_args": {"time_min_num_masks": 2,
    #                                                              "time_max_mask_per_n_frames": 25,
    #                                                              "time_mask_max_size": 20,
    #                                                              "freq_min_num_masks": 2,
    #                                                              "freq_mask_max_size": 5,
    #                                                              "freq_max_num_masks": 8},
    #                                             "final_dropout": 0.1,
    #                                             "att_weights_dropout": 0.1,
    #                                             "dropout": 0.1,
    #                                             "adjust_dropout": adjust_dropout,
    #                                             "model_dim": 512,
    #                                             "total_num_components": 240,
    #                                             "lst_cmp_cost": ([525184] * 4 + [201600] * 4 + [131456] * 8 + [525184] * 4) * 12}
    #
    #                             grad_score_opts = {
    #                                 "grad_score_update_steps": 1000,
    #                                 "grad_score_metric": grad_score_metric,
    #                                 "grad_update_beta": grad_update_beta
    #                             }
    #                             network_args["grad_score_opts"] = grad_score_opts
    #                             adaptation_opts = {
    #                                 "adaptation_global_step": [1220 * i for i in adaptation_global_epochs],
    #                                 "total_cost": 74165392,
    #                                 "rest_cost": 1451152,
    #                                 "lst_replace_pct": lst_replace_pct,
    #                                 "dict_module_cost": {
    #                                     "ff1": 525184,
    #                                     "conv": 201600,
    #                                     "mhsa": 131456,
    #                                     "ff2": 525184,
    #                                 }
    #                             }
    #                             network_args["adaptation_opts"] = adaptation_opts
    #                             component_dist = {
    #                                 "ff1": [4] * 12,
    #                                 "conv": [4] * 12,
    #                                 "mhsa": [8] * 12,
    #                                 "ff2": [4] * 12,
    #                             }
    #                             network_args["component_dist"] = component_dist
    #
    #                             config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
    #                                                            network_args=network_args)
    #                             num_layers_12_experiment_name = f"double_prune_total_num_components_240_freq_mask_50_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
    #                             recog_network_args = copy.deepcopy(network_args)
    #                             if num_layers_12_experiment_name in dict_recog_component_dist:
    #                                 recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
    #                             else:
    #                                 recog_component_dist = component_dist
    #                             lst_ff1_h_dims = [i * 512 for i in recog_component_dist["ff1"]]
    #                             lst_ff2_h_dims = [i * 512 for i in recog_component_dist["ff2"]]
    #                             lst_att_heads = recog_component_dist["mhsa"]
    #                             lst_channels = [i * 128 for i in recog_component_dist["conv"]]
    #                             recog_network_args["lst_ff1_h_dims"] = lst_ff1_h_dims
    #                             recog_network_args["lst_ff2_h_dims"] = lst_ff2_h_dims
    #                             recog_network_args["lst_att_heads"] = lst_att_heads
    #                             recog_network_args["lst_channels"] = lst_channels
    #                             total_num_components = 0
    #                             for k, v in recog_component_dist.items():
    #                                 total_num_components += sum(v)
    #                             recog_network_args["total_num_components"] = total_num_components
    #                             recog_network_args["lst_cmp_cost"] = [0] * total_num_components
    #
    #                             recog_config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
    #                                                                  network_args=recog_network_args)
    #                             experiments[num_layers_12_experiment_name] = get_returnn_configs(config, recog_config)
    #
    # for final_adapt_epochs in [90]:
    #     for pct in [0.15, 0.2, 0.25]:
    #         for num_iters in [1, 2, 4, 8]:
    #             for adjust_dropout in [True]:
    #                 if pct / num_iters > 0.007:
    #                     lst_replace_pct = [pct / num_iters] * num_iters
    #                     num_epochs_per_iter = final_adapt_epochs // num_iters
    #                     adaptation_global_epochs = list(
    #                         range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
    #                     if len(adaptation_global_epochs) > len(lst_replace_pct):
    #                         adaptation_global_epochs = adaptation_global_epochs[:num_iters]
    #                     assert len(adaptation_global_epochs) == len(lst_replace_pct)
    #
    #                     for grad_update_beta in [0.1]:
    #                         for grad_score_metric in ["first_taylor"]:
    #                             network_args = {"specaug_args": {"time_min_num_masks": 2,
    #                                                              "time_max_mask_per_n_frames": 25,
    #                                                              "time_mask_max_size": 20,
    #                                                              "freq_min_num_masks": 2,
    #                                                              "freq_mask_max_size": 5,
    #                                                              "freq_max_num_masks": 8},
    #                                             "final_dropout": 0.1,
    #                                             "att_weights_dropout": 0.1,
    #                                             "dropout": 0.1,
    #                                             "adjust_dropout": adjust_dropout,
    #                                             "model_dim": 512,
    #                                             "total_num_components": 240,
    #                                             "lst_cmp_cost": ([525184] * 4 + [201600] * 4 + [131456] * 8 + [525184] * 4) * 12}
    #
    #                             grad_score_opts = {
    #                                 "grad_score_update_steps": 1000,
    #                                 "grad_score_metric": grad_score_metric,
    #                                 "grad_update_beta": grad_update_beta
    #                             }
    #                             network_args["grad_score_opts"] = grad_score_opts
    #                             adaptation_opts = {
    #                                 "adaptation_global_step": [1220 * i for i in adaptation_global_epochs],
    #                                 "total_cost": 74165392,
    #                                 "rest_cost": 1451152,
    #                                 "lst_replace_pct": lst_replace_pct,
    #                                 "dict_module_cost": {
    #                                     "ff1": 525184,
    #                                     "conv": 201600,
    #                                     "mhsa": 131456,
    #                                     "ff2": 525184,
    #                                 }
    #                             }
    #                             network_args["adaptation_opts"] = adaptation_opts
    #                             component_dist = {
    #                                 "ff1": [4] * 12,
    #                                 "conv": [4] * 12,
    #                                 "mhsa": [8] * 12,
    #                                 "ff2": [4] * 12,
    #                             }
    #                             network_args["component_dist"] = component_dist
    #
    #                             config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
    #                                                            network_args=network_args)
    #                             num_layers_12_experiment_name = f"double_prune_total_num_components_240_freq_mask_50_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
    #                             recog_network_args = copy.deepcopy(network_args)
    #                             if num_layers_12_experiment_name in dict_recog_component_dist:
    #                                 recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
    #                             else:
    #                                 recog_component_dist = component_dist
    #                             lst_ff1_h_dims = [i * 512 for i in recog_component_dist["ff1"]]
    #                             lst_ff2_h_dims = [i * 512 for i in recog_component_dist["ff2"]]
    #                             lst_att_heads = recog_component_dist["mhsa"]
    #                             lst_channels = [i * 128 for i in recog_component_dist["conv"]]
    #                             recog_network_args["lst_ff1_h_dims"] = lst_ff1_h_dims
    #                             recog_network_args["lst_ff2_h_dims"] = lst_ff2_h_dims
    #                             recog_network_args["lst_att_heads"] = lst_att_heads
    #                             recog_network_args["lst_channels"] = lst_channels
    #                             total_num_components = 0
    #                             for k, v in recog_component_dist.items():
    #                                 total_num_components += sum(v)
    #                             recog_network_args["total_num_components"] = total_num_components
    #                             recog_network_args["lst_cmp_cost"] = [0] * total_num_components
    #
    #                             recog_config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
    #                                                                  network_args=recog_network_args)
    #                             experiments[num_layers_12_experiment_name] = get_returnn_configs(config, recog_config)
    #
    for final_adapt_epochs in [90, 60, 30]:
        for pct in [0.15]:
            for num_iters in [1, 4, 6]:
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
                            for grad_score_metric in ["first_taylor"]:
                                network_args = {"specaug_args": {"time_min_num_masks": 2,
                                                                 "time_max_mask_per_n_frames": 25,
                                                                 "time_mask_max_size": 20,
                                                                 "freq_min_num_masks": 2,
                                                                 "freq_mask_max_size": 5,
                                                                 "freq_max_num_masks": 8},
                                                "final_dropout": 0.1,
                                                "att_weights_dropout": 0.1,
                                                "dropout": 0.1,
                                                "adjust_dropout": adjust_dropout,
                                                "model_dim": 512,
                                                "total_num_components": 240,
                                                "lst_cmp_cost": ([525184] * 4 + [201600] * 4 + [131456] * 8 + [525184] * 4) * 12}

                                grad_score_opts = {
                                    "grad_score_update_steps": 1000,
                                    "grad_score_metric": grad_score_metric,
                                    "grad_update_beta": grad_update_beta
                                }
                                network_args["grad_score_opts"] = grad_score_opts
                                adaptation_opts = {
                                    "adaptation_global_step": [1220 * i for i in adaptation_global_epochs],
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
                                num_layers_12_experiment_name = f"double_prune_total_num_components_240_freq_mask_50_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
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
    #
    # for final_adapt_epochs in [90]:
    #     for pct in [0.15]:
    #         for num_iters in [1]:
    #             for adjust_dropout in [True]:
    #                 if pct / num_iters > 0.007:
    #                     lst_replace_pct = [pct / num_iters] * num_iters
    #                     num_epochs_per_iter = final_adapt_epochs // num_iters
    #                     adaptation_global_epochs = list(
    #                         range(num_epochs_per_iter, final_adapt_epochs + num_epochs_per_iter, num_epochs_per_iter))
    #                     if len(adaptation_global_epochs) > len(lst_replace_pct):
    #                         adaptation_global_epochs = adaptation_global_epochs[:num_iters]
    #                     assert len(adaptation_global_epochs) == len(lst_replace_pct)
    #
    #                     for grad_update_beta in [0.1]:
    #                         for grad_score_metric in ["first_taylor"]:
    #                             for num_chunks in [4]:
    #                                 total_num_components = (3 * num_chunks + 8) * 12
    #                                 network_args = {"specaug_args": {"time_min_num_masks": 2,
    #                                                                  "time_max_mask_per_n_frames": 25,
    #                                                                  "time_mask_max_size": 20,
    #                                                                  "freq_min_num_masks": 2,
    #                                                                  "freq_mask_max_size": 5,
    #                                                                  "freq_max_num_masks": 8},
    #                                                 "final_dropout": 0.1,
    #                                                 "att_weights_dropout": 0.1,
    #                                                 "dropout": 0.1,
    #                                                 "adjust_dropout": adjust_dropout,
    #                                                 "model_dim": 512,
    #                                                 "total_num_components": total_num_components,
    #                                                 "lst_cmp_cost": ([525184 * 4 // num_chunks] * num_chunks + [
    #                                                     201600 * 4 // num_chunks] * num_chunks + [131456] * 8 + [
    #                                                                      525184 * 4 // num_chunks] * num_chunks) * 12}
    #
    #                                 grad_score_opts = {
    #                                     "grad_score_update_steps": 1000,
    #                                     "grad_score_metric": grad_score_metric,
    #                                     "grad_update_beta": grad_update_beta
    #                                 }
    #                                 network_args["grad_score_opts"] = grad_score_opts
    #                                 adaptation_opts = {
    #                                     "adaptation_global_step": [1220 * i for i in adaptation_global_epochs],
    #                                     "total_cost": 74165392,
    #                                     "rest_cost": 1451152,
    #                                     "lst_replace_pct": lst_replace_pct,
    #                                     "dict_module_cost": {
    #                                         "ff1": 525184 * 4 // num_chunks,
    #                                         "conv": 201600 * 4 // num_chunks,
    #                                         "mhsa": 131456,
    #                                         "ff2": 525184 * 4 // num_chunks,
    #                                     }
    #                                 }
    #                                 network_args["adaptation_opts"] = adaptation_opts
    #                                 component_dist = {
    #                                     "ff1": [num_chunks] * 12,
    #                                     "conv": [num_chunks] * 12,
    #                                     "mhsa": [8] * 12,
    #                                     "ff2": [num_chunks] * 12,
    #                                 }
    #                                 network_args["component_dist"] = component_dist
    #
    #                                 config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
    #                                                                network_args=network_args)
    #                                 num_layers_12_experiment_name = f"double_prune_total_num_components_{total_num_components}_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
    #                                 recog_network_args = copy.deepcopy(network_args)
    #                                 if num_layers_12_experiment_name in dict_recog_component_dist:
    #                                     recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
    #                                 else:
    #                                     recog_component_dist = component_dist
    #                                 lst_ff1_h_dims = [i * ((512 * 4) // num_chunks) for i in
    #                                                   recog_component_dist["ff1"]]
    #                                 lst_ff2_h_dims = [i * ((512 * 4) // num_chunks) for i in
    #                                                   recog_component_dist["ff2"]]
    #                                 lst_channels = [i * ((128 * 4) // num_chunks) for i in recog_component_dist["conv"]]
    #                                 lst_att_heads = recog_component_dist["mhsa"]
    #                                 recog_network_args["lst_ff1_h_dims"] = lst_ff1_h_dims
    #                                 recog_network_args["lst_ff2_h_dims"] = lst_ff2_h_dims
    #                                 recog_network_args["lst_att_heads"] = lst_att_heads
    #                                 recog_network_args["lst_channels"] = lst_channels
    #                                 total_num_components = 0
    #                                 for k, v in recog_component_dist.items():
    #                                     total_num_components += sum(v)
    #                                 recog_network_args["total_num_components"] = total_num_components
    #                                 recog_network_args["lst_cmp_cost"] = [0] * total_num_components
    #
    #                                 recog_config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
    #                                                                      network_args=recog_network_args)
    #                                 experiments[num_layers_12_experiment_name] = get_returnn_configs(config, recog_config)

    return experiments