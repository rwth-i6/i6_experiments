from sisyphus import *

import os, sys
import copy
import itertools
from collections import ChainMap
from typing import List

from i6_core import rasr
from i6_core import returnn

from i6_experiments.users.jxu.models.conformer import conformer_network
from i6_experiments.users.jxu.experiments.hybrid.switchboard.baseline_args_jingjing import get_returnn_configs_jingjing, get_nn_args, RECUSRION_LIMIT
from i6_experiments.users.jxu.experiments.hybrid.switchboard.reduced_dim import network as reduced
from i6_experiments.users.jxu.experiments.hybrid.switchboard.data import get_corpus_data_inputs_newcv
from i6_experiments.users.jxu.experiments.hybrid.switchboard.default_tools import RASR_BINARY_PATH, RETURNN_ROOT
from i6_experiments.users.jxu.experiments.hybrid.switchboard.conformer_baseline import run_gmm_system_from_common, run_hybrid_baseline_jingjing

from . import default_tools as tools

from i6_experiments.common.setups.rasr.util import RasrSteps, HybridArgs
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

sys.setrecursionlimit(2500)

fname = os.path.split(__file__)[1].split('.')[0]
# gs.ALIAS_AND_OUTPUT_SUBDIR = fname

REC_LIMIT_CODE_TMPLT = """
import sys
sys.setrecursionlimit({rec_limit})
"""

TRAIN_CV_KEY = "_".join(["switchboard.train", "switchboard.cv"])

def recursion_limit_code(rec_limit=3000):
    return REC_LIMIT_CODE_TMPLT.format(rec_limit=rec_limit)

def init_hybrid_system_jingjing():
    gmm_system = run_gmm_system_from_common()
    rasr_init_args = copy.deepcopy(gmm_system.rasr_init_args)

    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    ) = get_corpus_data_inputs_newcv(gmm_system)

    hybrid_nn_system = tools.SwbSystemWithDefaultTools()
    hybrid_nn_system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=nn_train_data_inputs,
        cv_data=nn_cv_data_inputs,
        devtrain_data=nn_devtrain_data_inputs,
        dev_data=nn_dev_data_inputs,
        test_data=nn_test_data_inputs,
        train_cv_pairing=[tuple(["switchboard.train", "switchboard.cv"])],
    )
    return hybrid_nn_system

def nn_steps_jingjing():
    nn_args = get_nn_args(num_epochs=260, peak_lr=2e-3)
    nn_steps = RasrSteps()
    nn_steps.add_step("nn", nn_args)
    return nn_steps


def baseline():
    hybrid_nn_system = init_hybrid_system_jingjing()
    hybrid_nn_system.run(nn_steps_jingjing())
    return hybrid_nn_system

default_conformer_args = dict(
    num_blocks=12,
    macron_on_conv=True,
    layernorm_replace_batchnorm=True,
    ln_on_source_linear=True,
    MLP_on_output=True,
    tconv_act="swish",
    enc_key_dim=384,
    ff_dim=1536,
    conv_kernel_size=8,
    conv_kernel_size_2=8,
    att_num_heads=6,
    ce_loss_ops={"focal_loss_factor": 2},
    ff_init={'class': 'VarianceScaling', 'distribution': 'uniform', 'mode': 'fan_in', 'scale': 0.78},
    # iterated_loss_layers=[4, 8]
)

def get_returnn_config_custom(
    num_inputs: int,
    num_outputs: int,
    evaluation_epochs: List[int],
    num_epochs: int = 260,
    recognition=False,
    extra_exps=False,
    peak_lr=2e-3,
    min_lr=None,
    const_lr=False,
    learning_rates=None,
    network_args=None,
    config_updates=None,
    special_network=None,
    extend_recursion_limit=True,
):
    # ******************** blstm base ********************
    network_args = network_args or {}

    base_config = {
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "sparse": True},
        },
    }
    base_post_config = {
        "use_tensorflow": True,
        # "debug_print_layer_output_template": True,
        # "log_batch_size": True,
        # "tf_log_memory_usage": True,
        "cache_size": "0",
    }
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }

    if not special_network:
        kwargs = ChainMap(network_args, default_conformer_args)
        network = conformer_network(**kwargs)
    else:
        network = copy.deepcopy(special_network)

    from i6_experiments.users.jxu.spec_aug.legacy_specaug_jingjing import (
        specaug_layer_jingjing,
        get_funcs_jingjing,
    )

    prolog_jingjing = get_funcs_jingjing()
    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "batch_size": 14000,
            "chunking": "500:250",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
        }
    )
    if peak_lr or learning_rates:
        import numpy as np
        if peak_lr and not min_lr:
            min_lr = peak_lr / 10
        conformer_base_config.update({
            "learning_rates": list(np.linspace(min_lr, peak_lr, 100))
                + list(np.linspace(peak_lr, min_lr, 100))
                + list(np.linspace(min_lr, 1e-8, 60)) if not learning_rates else learning_rates,
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            # "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 3,
            "newbob_multi_update_interval": 1,
        })
    elif const_lr:
        assert not peak_lr
        conformer_base_config["learning_rate"] = const_lr
        conformer_base_config["learning_rate_control"] = "constant"
    else:
        raise AssertionError("Either peak_lr or const_lr must be set")
    if config_updates:
        conformer_base_config.update(config_updates)
    conformer_jingjing_config = copy.deepcopy(conformer_base_config)

    def make_returnn_config(
        config,
        python_prolog,
        staged_network_dict=None,
    ):
        if recognition:
            rec_network = copy.deepcopy(network)
            output_source_layer = rec_network["output"]["from"]
            if not isinstance(output_source_layer, str):
                output_source_layer = output_source_layer[0]
            rec_network["output"] = {
                "class": "linear",
                "activation": "log_softmax",
                # "from": ["MLP_output"],
                "from": [output_source_layer],
                "n_out": num_outputs,
                # "is_output_layer": True,
            }
            rec_network["source"] = {'class': 'copy', 'from': 'data'}
            config["network"] = rec_network
        else:
            train_network = copy.deepcopy(network)
            train_network["source"] = specaug_layer_jingjing(in_layer=["data"])
            config["network"] = train_network
        
        return returnn.ReturnnConfig(
            config=config,
            post_config=base_post_config,
            staged_network_dict=staged_network_dict if not recognition else None,
            hash_full_python_code=True,
            python_prolog=python_prolog if not recognition else None,
            python_epilog=recursion_limit_code(
                max(3000, 250 * kwargs["num_blocks"])
            ) if extend_recursion_limit else None,
            pprint_kwargs={"sort_dicts": False},
        )

    conformer_jingjing_returnn_config = make_returnn_config(
        conformer_jingjing_config,
        staged_network_dict=None,
        python_prolog=prolog_jingjing,
    )

    return conformer_jingjing_returnn_config

def get_returnn_config_swb(
    recognition,
    const_lr,
    network_args=None,
    peak_lr=None,
    learning_rates=None,
):
    return get_returnn_config_custom(
        num_inputs=40,
        num_outputs=9001,
        evaluation_epochs=[12, 24, 48, 120, 180, 240, 260],
        recognition=recognition,
        network_args=network_args,
        peak_lr=peak_lr,
        const_lr=const_lr,
    )

def compare_to_reduced(network, base=reduced):
    if set(network.keys()) != set(base.keys()):
        net_keys = set(network.keys())
        base_keys = set(base.keys())
        print("Keys not matching")
        print("Only network keys:")
        print(net_keys - base_keys)
        print("Only base keys:")
        print(base_keys - net_keys)
        quit()
    for key in network:
        if network[key] == base[key]:
            continue
        print(key)
        print(network[key])
        print(base[key])
        quit()

def compare_to_base(config, recog=False):
    base_nn_args = get_nn_args(num_epochs=260, peak_lr=2e-3)

    base_config = base_nn_args.returnn_training_configs["conformer_jingjing"] if not recog else base_nn_args.returnn_recognition_configs["conformer_jingjing"]

    if set(base_config.config.keys()) != set(config.config.keys()):
        print("Config keys:")
        print(config.config.keys())
        print("Base keys:")
        print(base_config.config.keys())
        quit()
    
    for key in config.config:
        if config.config[key] == base_config.config[key]:
            continue
        if key == "network":
            compare_to_reduced(config.config[key], base_config.config[key])
        print(key)
        print(config.config[key])
        print(base_config.config[key])
        quit()

