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
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_conformer.adapt_based_on_gradient_ranking_finer_granul_random_init import \
        get_train_serializer, get_recog_serializer, get_prior_serializer
    if variant == ConfigVariant.TRAIN:
        return get_train_serializer(model_config)
    if variant == ConfigVariant.PRIOR:
        return get_prior_serializer(model_config)
    if variant == ConfigVariant.RECOG:
        return get_recog_serializer(model_config)
    raise NotImplementedError


def returnn_config_generator(train_data_config: dict, dev_data_config: dict, peak_lr: float) -> dict:
    from i6_experiments.users.jxu.experiments.ctc.tedlium2.pytorch_networks.neural_block.dynamic_adaptable_conformer.adapt_based_on_gradient_ranking_finer_granul_random_init import \
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
        "random_prune_total_num_components_13896_beta_0.1_gradient_update_step_1000":
            {'ff1': [197, 468, 371, 330, 320, 334, 350, 399, 428, 482, 641, 474],
             'ff2': [178, 328, 226, 198, 182, 203, 256, 324, 398, 444, 392, 385],
             'conv': [146, 292, 402, 447, 513, 520, 553, 649, 696, 735, 753, 753],
             'mhsa': [7, 12, 9, 6, 6, 6, 6, 6, 6, 6, 6, 2]},
        "random_prune_total_num_components_13896_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [207, 515, 386, 359, 349, 333, 378, 406, 438, 539, 603, 422],
             'ff2': [223, 338, 296, 226, 226, 267, 273, 346, 408, 485, 387, 382],
             'conv': [140, 300, 389, 419, 477, 501, 562, 631, 655, 706, 715, 708],
             'mhsa': [6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0]},

        "random_prune_total_num_components_1224_beta_0.1_gradient_update_step_1000":
            {'ff1': [26, 33, 32, 32, 32, 31, 31, 32, 32, 32, 53, 63],
             'ff2': [15, 32, 25, 12, 1, 0, 2, 22, 32, 32, 32, 32],
             'conv': [13, 20, 45, 46, 48, 48, 49, 52, 61, 64, 64, 64], 'mhsa': [7, 12, 9, 7, 6, 6, 6, 6, 6, 6, 6, 6]},
        "random_prune_total_num_components_1224_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [27, 34, 32, 32, 31, 31, 31, 32, 32, 33, 64, 62],
             'ff2': [21, 32, 27, 14, 4, 6, 6, 30, 32, 35, 32, 32],
             'conv': [11, 22, 35, 46, 47, 48, 48, 53, 62, 64, 64, 64], 'mhsa': [6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 1, 0]},

        "random_prune_total_num_components_216_beta_0.1_gradient_update_step_1000":
            {'ff1': [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 8], 'ff2': [1, 4, 4, 2, 0, 0, 0, 4, 4, 4, 4, 4],
             'conv': [2, 2, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8], 'mhsa': [7, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 5]},
        "random_prune_total_num_components_216_beta_0.1_first_taylor_update_step_1000":
            {'ff1': [2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4], 'ff2': [3, 4, 4, 3, 0, 0, 1, 4, 4, 7, 4, 4],
             'conv': [1, 2, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8], 'mhsa': [6, 12, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0]}
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
                                    for num_chunks in [4, 32, 384]:
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
                                        num_layers_12_experiment_name = f"random_prune_total_num_components_{total_num_components}_beta_{grad_update_beta}_{grad_score_metric}_update_step_{grad_score_update_steps}"
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
