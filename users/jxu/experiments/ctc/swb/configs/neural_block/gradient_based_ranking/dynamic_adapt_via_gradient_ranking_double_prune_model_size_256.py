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
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.1_iters_1_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4], 'ff2': [4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4],
             'conv': [4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 'mhsa': [8, 8, 5, 7, 4, 4, 4, 4, 4, 4, 4, 4]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.1_iters_2_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 4, 4, 4, 4, 4, 0, 1, 2, 1, 2, 0],
             'conv': [4, 5, 6, 6, 6, 7, 6, 6, 6, 6, 6, 3], 'mhsa': [7, 8, 6, 7, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.1_iters_4_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], 'ff2': [4, 4, 2, 2, 4, 2, 4, 1, 0, 4, 4, 0],
             'conv': [2, 4, 5, 4, 7, 6, 6, 8, 8, 9, 10, 1], 'mhsa': [9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        # "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.1_iters_8_0.0004":

        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.15_iters_1_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4], 'ff2': [4, 4, 2, 1, 0, 0, 0, 4, 4, 4, 4, 4], 
        'conv': [3, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8], 'mhsa': [8, 8, 7, 4, 4, 4, 1, 4, 4, 4, 4, 1]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.15_iters_2_0.0004":
        {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 4, 4, 4, 0, 0, 0, 1, 4, 4, 4, 0], 
        'conv': [4, 5, 6, 6, 8, 7, 6, 9, 9, 10, 10, 4], 'mhsa': [8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.15_iters_4_0.0004":
         {'ff1': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 3, 3, 3, 1, 3, 4, 4, 0, 0, 2, 0], 
         'conv': [2, 6, 3, 6, 3, 5, 7, 9, 11, 12, 8, 1], 'mhsa': [9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.15_iters_8_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 4, 0, 2, 0, 3, 0, 4, 4, 4, 2, 0], 
        'conv': [2, 6, 4, 3, 7, 8, 8, 7, 7, 9, 10, 1], 'mhsa': [10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.15_iters_16_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3], 'ff2': [4, 3, 4, 2, 0, 0, 4, 4, 0, 4, 4, 0], 
        'conv': [2, 5, 3, 6, 6, 7, 8, 8, 6, 9, 7, 1], 'mhsa': [9, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.2_iters_1_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 0, 2, 4, 4, 4, 4], 'ff2': [4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 4, 4], 
        'conv': [2, 6, 6, 6, 8, 8, 7, 8, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 8, 8, 4, 5, 5, 5, 5, 3]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.2_iters_2_0.0004":
        {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3], 'ff2': [4, 4, 4, 2, 4, 3, 0, 0, 0, 2, 3, 0], 
        'conv': [2, 6, 8, 8, 9, 8, 6, 6, 7, 7, 6, 0], 'mhsa': [8, 8, 8, 8, 7, 6, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.2_iters_4_0.0004":
         {'ff1': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 3, 2, 2, 4, 0, 0, 0, 0, 4, 0, 0], 
         'conv': [2, 4, 3, 9, 7, 8, 9, 8, 11, 8, 9, 1], 'mhsa': [11, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.2_iters_8_0.0004":
         {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 4, 4, 0, 1, 3, 0, 4, 0, 4, 2, 0], 
         'conv': [1, 4, 4, 7, 3, 6, 6, 10, 7, 7, 10, 0], 'mhsa': [13, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.2_iters_16_0.0004":
         {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 0], 
         'conv': [1, 5, 2, 6, 6, 3, 8, 6, 6, 9, 7, 3], 'mhsa': [9, 6, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0]},
        
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.25_iters_1_0.0004":
        {'ff1': [4, 4, 4, 3, 4, 1, 0, 4, 4, 7, 8, 6], 'ff2': [4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 4, 6], 
        'conv': [3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 4, 4, 4, 4, 7, 4, 5, 2]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.25_iters_2_0.0004":
         {'ff1': [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [0, 2, 2, 2, 2, 1, 0, 0, 4, 4, 3, 0], 
         'conv': [3, 4, 5, 7, 8, 9, 9, 10, 11, 12, 11, 4], 'mhsa': [8, 8, 8, 7, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.25_iters_4_0.0004":
        {'ff1': [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 3, 3, 0, 1, 4, 0, 0, 0, 4, 0, 0], 
        'conv': [2, 4, 4, 6, 2, 7, 7, 11, 11, 16, 14, 1], 'mhsa': [13, 8, 4, 4, 4, 4, 4, 4, 4, 4, 1, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.25_iters_8_0.0004":
        {'ff1': [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 1, 0, 2, 3, 0, 4, 0, 4, 4, 4, 0], 
        'conv': [1, 6, 2, 2, 6, 6, 6, 10, 7, 10, 9, 2], 'mhsa': [10, 7, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.25_iters_16_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], 'ff2': [4, 2, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0], 
        'conv': [1, 2, 5, 7, 3, 4, 10, 6, 8, 9, 6, 2], 'mhsa': [8, 7, 5, 4, 4, 4, 4, 4, 4, 4, 6, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_50_pct_0.25_iters_32_0.0004":
        {'ff1': [4, 4, 3, 3, 4, 4, 3, 4, 4, 4, 4, 4], 'ff2': [0, 2, 2, 1, 0, 0, 0, 0, 0, 4, 4, 0], 
        'conv': [7, 2, 4, 2, 5, 5, 8, 8, 6, 9, 7, 2], 'mhsa': [13, 6, 4, 4, 4, 4, 4, 4, 6, 8, 13, 0]},

        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.1_iters_1_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [4, 4, 4, 4, 4, 0, 0, 0, 2, 4, 4, 4],
             'conv': [2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 'mhsa': [7, 8, 8, 5, 4, 5, 4, 4, 4, 4, 4, 4]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.1_iters_2_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], 'ff2': [3, 4, 4, 4, 4, 4, 3, 2, 4, 4, 4, 0],
             'conv': [2, 4, 4, 6, 5, 6, 7, 8, 9, 10, 10, 2], 'mhsa': [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        # "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.1_iters_4_0.0004":
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.1_iters_8_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 4, 4, 4, 4, 2, 0, 0, 4, 4, 4, 0],
             'conv': [1, 5, 4, 5, 7, 5, 7, 7, 8, 9, 9, 1], 'mhsa': [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},

        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.15_iters_1_0.0004":
        {'ff1': [4, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'ff2': [0, 4, 4, 4, 1, 0, 4, 4, 4, 3, 0, 1], 
        'conv': [2, 6, 6, 6, 6, 6, 6, 8, 8, 8, 7, 6], 'mhsa': [8, 8, 8, 4, 7, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.15_iters_2_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], 'ff2': [0, 4, 4, 4, 4, 1, 0, 0, 4, 4, 4, 0], 
        'conv': [1, 4, 4, 6, 6, 7, 6, 10, 10, 11, 11, 3], 'mhsa': [7, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.15_iters_4_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 2], 'ff2': [4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0], 
        'conv': [1, 4, 5, 4, 4, 6, 6, 7, 8, 10, 13, 2], 'mhsa': [3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 3, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.15_iters_8_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 0], 'ff2': [4, 4, 4, 3, 4, 3, 0, 0, 0, 4, 2, 0], 
        'conv': [1, 3, 3, 6, 7, 6, 6, 7, 8, 10, 10, 1], 'mhsa': [9, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.15_iters_16_0.0004":
        {'ff1': [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 4, 3, 1, 0, 3, 0, 4, 4, 4, 1, 0],
        'conv': [1, 4, 5, 8, 6, 5, 4, 8, 10, 12, 10, 0], 'mhsa': [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.2_iters_1_0.0004":
        {'ff1': [4, 8, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4], 'ff2': [4, 4, 3, 4, 0, 0, 1, 2, 4, 2, 0, 0], 
        'conv': [2, 6, 6, 6, 8, 6, 7, 8, 8, 8, 8, 6], 'mhsa': [8, 8, 8, 5, 4, 6, 4, 5, 8, 4, 2, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.2_iters_2_0.0004":
        {'ff1': [0, 6, 4, 4, 4, 4, 4, 4, 4, 5, 4, 0], 'ff2': [2, 4, 4, 4, 4, 4, 0, 0, 3, 4, 0, 0], 
        'conv': [2, 4, 6, 6, 7, 8, 7, 10, 10, 11, 10, 3], 'mhsa': [8, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.2_iters_4_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 7, 0], 'ff2': [3, 4, 4, 4, 0, 0, 0, 0, 4, 4, 1, 0], 
        'conv': [1, 5, 4, 6, 4, 4, 6, 6, 8, 12, 13, 2], 'mhsa': [8, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.2_iters_8_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 0], 'ff2': [4, 4, 4, 3, 0, 2, 0, 0, 4, 4, 3, 0], 
        'conv': [1, 3, 3, 4, 7, 2, 4, 7, 8, 13, 14, 1], 'mhsa': [8, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.2_iters_16_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 2, 4, 0, 4, 3, 0, 4, 4, 4, 0, 0], 
        'conv': [4, 3, 2, 4, 3, 8, 4, 9, 8, 15, 14, 0], 'mhsa': [9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 0]},
        # "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.2_iters_32_0.0004":
        
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_1_0.0004":
        {'ff1': [0, 8, 4, 4, 4, 4, 3, 4, 4, 5, 8, 4], 'ff2': [0, 4, 0, 2, 0, 0, 0, 0, 3, 4, 4, 4],
         'conv': [1, 5, 6, 6, 7, 6, 6, 8, 8, 8, 8, 8], 'mhsa': [8, 8, 8, 8, 6, 8, 4, 5, 8, 5, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_2_0.0004":
        {'ff1': [0, 7, 4, 4, 4, 4, 4, 4, 4, 6, 8, 0], 'ff2': [0, 4, 4, 3, 0, 0, 0, 0, 4, 4, 1, 0], 
        'conv': [2, 4, 6, 6, 6, 7, 6, 10, 10, 10, 12, 4], 'mhsa': [8, 7, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_4_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 5, 8, 5, 0], 'ff2': [0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 0, 0], 
        'conv': [1, 4, 4, 7, 6, 5, 5, 8, 10, 12, 13, 2], 'mhsa': [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_8_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 0], 'ff2': [4, 2, 4, 3, 0, 2, 0, 0, 4, 4, 2, 0], 
        'conv': [1, 4, 4, 4, 3, 5, 6, 9, 7, 12, 11, 2], 'mhsa': [7, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_16_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0], 'ff2': [4, 2, 1, 0, 4, 3, 0, 0, 4, 4, 0, 0], 
        'conv': [1, 4, 4, 6, 4, 6, 4, 7, 10, 12, 15, 0], 'mhsa': [12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_32_0.0004":
        {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 5, 0], 'ff2': [4, 3, 2, 0, 0, 4, 0, 4, 4, 4, 0, 0], 
        'conv': [1, 1, 6, 4, 4, 8, 6, 8, 9, 17, 9, 0], 'mhsa': [10, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0]},
        # "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_75_pct_0.25_iters_32_0.0004":
        #     {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 5, 0], 'ff2': [4, 3, 2, 0, 0, 4, 0, 4, 4, 4, 0, 0],
        #      'conv': [1, 1, 6, 4, 4, 8, 6, 8, 9, 17, 9, 0], 'mhsa': [10, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0]},

        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.15_iters_1_0.0004":
            {'ff1': [0, 8, 5, 4, 4, 4, 4, 4, 4, 5, 5, 4], 'ff2': [0, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 0],
             'conv': [1, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 7], 'mhsa': [4, 8, 5, 4, 4, 4, 0, 0, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.15_iters_2_0.0004":
            {'ff1': [2, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], 'ff2': [3, 4, 4, 4, 2, 0, 1, 2, 4, 4, 4, 0],
             'conv': [2, 6, 6, 6, 7, 7, 6, 8, 7, 10, 9, 2], 'mhsa': [0, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.15_iters_4_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 0], 'ff2': [4, 4, 4, 4, 4, 2, 0, 0, 0, 4, 0, 0],
             'conv': [1, 4, 6, 6, 7, 6, 6, 7, 11, 10, 9, 0], 'mhsa': [6, 8, 7, 4, 4, 4, 0, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.15_iters_8_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 0], 'ff2': [4, 4, 3, 3, 3, 4, 1, 0, 2, 4, 3, 0],
             'conv': [1, 3, 3, 5, 6, 5, 5, 7, 9, 10, 10, 1], 'mhsa': [7, 6, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.15_iters_16_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [4, 4, 0, 0, 4, 4, 0, 4, 4, 4, 0, 0],
             'conv': [1, 4, 3, 6, 8, 8, 6, 8, 7, 9, 8, 0], 'mhsa': [8, 7, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0]},

        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.2_iters_1_0.0004":
            {'ff1': [0, 8, 4, 4, 4, 4, 4, 4, 5, 8, 8, 5], 'ff2': [0, 4, 4, 3, 2, 0, 4, 4, 4, 4, 4, 1],
             'conv': [0, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8], 'mhsa': [4, 8, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.2_iters_2_0.0004":
            {'ff1': [2, 8, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0], 'ff2': [2, 4, 4, 4, 3, 0, 0, 0, 3, 4, 3, 0],
             'conv': [2, 4, 5, 6, 6, 6, 6, 9, 10, 10, 10, 3], 'mhsa': [2, 8, 7, 4, 4, 4, 2, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.2_iters_4_0.0004":
            {'ff1': [4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], 'ff2': [0, 4, 4, 4, 4, 0, 0, 0, 2, 4, 2, 0],
             'conv': [0, 5, 6, 6, 7, 4, 4, 8, 9, 9, 9, 3], 'mhsa': [7, 10, 8, 4, 4, 4, 2, 4, 4, 4, 4, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.2_iters_8_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 6, 0], 'ff2': [2, 3, 3, 3, 0, 4, 0, 4, 0, 4, 0, 0],
             'conv': [1, 5, 3, 6, 4, 5, 5, 7, 10, 10, 12, 3], 'mhsa': [3, 6, 7, 4, 4, 4, 4, 4, 4, 4, 3, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.2_iters_16_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3], 'ff2': [4, 4, 0, 0, 4, 4, 0, 4, 4, 4, 4, 0],
             'conv': [1, 5, 2, 8, 3, 6, 7, 5, 6, 9, 12, 2], 'mhsa': [8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0]},

        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.25_iters_1_0.0004":
            {'ff1': [0, 8, 7, 4, 4, 4, 4, 4, 7, 8, 8, 4], 'ff2': [0, 5, 4, 3, 3, 0, 1, 4, 4, 7, 4, 0],
             'conv': [0, 4, 5, 5, 6, 6, 6, 8, 8, 8, 8, 8], 'mhsa': [4, 8, 4, 4, 0, 0, 0, 0, 4, 4, 1, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.25_iters_2_0.0004":
            {'ff1': [0, 8, 4, 4, 4, 4, 4, 4, 5, 7, 6, 0], 'ff2': [0, 4, 4, 4, 0, 0, 0, 0, 2, 4, 0, 0],
             'conv': [0, 4, 6, 5, 6, 6, 7, 10, 12, 12, 12, 2], 'mhsa': [3, 8, 8, 4, 4, 4, 4, 4, 4, 4, 2, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.25_iters_4_0.0004":
            {'ff1': [0, 5, 4, 4, 4, 4, 4, 4, 4, 8, 5, 0], 'ff2': [4, 4, 4, 4, 2, 0, 0, 0, 2, 4, 0, 0],
             'conv': [1, 3, 6, 7, 6, 6, 6, 8, 10, 11, 11, 2], 'mhsa': [0, 8, 8, 4, 4, 3, 0, 4, 5, 4, 0, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.25_iters_8_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 0], 'ff2': [4, 3, 3, 3, 2, 4, 0, 0, 0, 4, 0, 0],
             'conv': [1, 3, 4, 6, 4, 4, 4, 6, 8, 10, 11, 3], 'mhsa': [5, 7, 8, 4, 4, 4, 3, 4, 4, 4, 0, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.25_iters_16_0.0004":
            {'ff1': [5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0], 'ff2': [4, 4, 2, 0, 4, 0, 0, 4, 4, 4, 1, 0],
             'conv': [1, 5, 4, 5, 4, 6, 5, 7, 8, 5, 8, 0], 'mhsa': [8, 7, 5, 7, 4, 4, 4, 4, 4, 6, 6, 0]},
        "double_prune_total_num_components_192_adjust_dropout_True_adapt_epochs_90_pct_0.25_iters_32_0.0004":
            {'ff1': [4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 5, 0], 'ff2': [4, 3, 2, 0, 0, 4, 0, 4, 4, 4, 0, 0],
             'conv': [1, 1, 6, 4, 4, 8, 6, 8, 9, 17, 9, 0], 'mhsa': [10, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0]}
    }

    for final_adapt_epochs in [50, 60, 75, 90]:
    # for final_adapt_epochs in [50]:
        for pct in [0.1, 0.15, 0.2, 0.25]:
        # for pct in [0.15]:
            for num_iters in [1, 2, 4, 8, 16, 32]:
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

                        for grad_update_beta in [0.1]:
                            for grad_score_metric in ["first_taylor"]:
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
                                                "model_dim": 256,
                                                "total_num_components": 192,
                                                "lst_cmp_cost": ([16440 * 8] * 4 + [6456 * 8] * 4 + [65920] * 4 + [
                                                    16440 * 8] * 4) * 12}

                                grad_score_opts = {
                                    "grad_score_update_steps": 1000,
                                    "grad_score_metric": grad_score_metric,
                                    "grad_update_beta": grad_update_beta
                                }
                                network_args["grad_score_opts"] = grad_score_opts
                                adaptation_opts = {
                                    "adaptation_global_step": [1200 * i for i in adaptation_global_epochs],
                                    "total_cost": 19025647,
                                    "rest_cost": 772751,
                                    "lst_replace_pct": lst_replace_pct,
                                    "dict_module_cost": {
                                        "ff1": 16440 * 8,
                                        "conv": 6456 * 8,
                                        "mhsa": 65920,
                                        "ff2": 16440 * 8,
                                    }
                                }
                                network_args["adaptation_opts"] = adaptation_opts
                                component_dist = {
                                    "ff1": [4] * 12,
                                    "conv": [4] * 12,
                                    "mhsa": [4] * 12,
                                    "ff2": [4] * 12,
                                }
                                network_args["component_dist"] = component_dist

                                config = get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
                                                               network_args=network_args)
                                str_adapt_epochs = "_".join([str(i) for i in adaptation_global_epochs])
                                repeat_times = lst_replace_pct.count(lst_replace_pct[0])
                                str_replace_pct = f"{str(lst_replace_pct[0])}_{repeat_times}"
                                # str_replace_pct = "_".join([str(i) for i in lst_replace_pct])
                                num_layers_12_experiment_name = f"double_prune_total_num_components_192_adjust_dropout_{adjust_dropout}_adapt_epochs_{final_adapt_epochs}_pct_{pct}_iters_{num_iters}_{peak_lr}"
                                recog_network_args = copy.deepcopy(network_args)
                                # assert num_layers_12_experiment_name in dict_recog_component_dist
                                if num_layers_12_experiment_name in dict_recog_component_dist:
                                    recog_component_dist = dict_recog_component_dist[num_layers_12_experiment_name]
                                else:
                                    recog_component_dist = component_dist
                                lst_ff1_h_dims = [i * 256 for i in recog_component_dist["ff1"]]
                                lst_ff2_h_dims = [i * 256 for i in recog_component_dist["ff2"]]
                                lst_att_heads = recog_component_dist["mhsa"]
                                lst_channels = [i * 64 for i in recog_component_dist["conv"]]
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