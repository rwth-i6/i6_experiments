__all__ = ["get_conformer_encoder"]

import copy
import dataclasses
import itertools
import typing

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common import oclr, returnn_time_tag
from ...setups.common.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)

from ...setups.fh.network import conformer
from ...setups.fh.factored import PhonemeStateClasses, PhoneticContext, RasrStateTying
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING,
    CONF_SA_CONFIG,
    RETURNN_PYTHON_TF15,
)

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_TF15)
RETURNN_PYTHON_EXE.hash_override = "FH_RETURNN_PYTHON_EXE"

def get_conformer_encoder(
    *,
    n_cart_out: int,
    num_inputs: int,
    conf_model_dim: int = 512,
    num_epochs: int = 300,
    lr: str = "v8",
)-> returnn.ReturnnConfig:
    # ******************** HY Init ********************

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=CONF_CHUNKING,
        label_smoothing=0.2,
        leave_cart_output=True,
        focal_loss_factor=2.0,
        num_classes=n_cart_out,
        num_inputs=num_inputs,
        time_tag_name=time_tag_name,
    )
    network = network_builder.network

    base_config = {
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "num_inputs": num_inputs,
        "batch_size": 10000,
        "use_tensorflow": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": CONF_CHUNKING,
        "optimizer": {"class": "nadam"},
        "gradient_clip": 0,
        "gradient_noise": 0.1,
        "optimizer_epsilon": 1e-08,
        "network": network,
        "extern_data": {
            "data": {
                "dim": 40,
                "shape": (None, 40),
                "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)},
            },
            "classes": {
                "dim": n_cart_out,
                "shape": (None,),
                "sparse": True,
                "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)},
            },
        },
    }
    keep_epochs = [280, num_epochs]
    base_post_config = {
        "cleanup_old_models": {
            "keep_best_n": 3,
            "keep": keep_epochs,
        },
    }
    returnn_config = returnn.ReturnnConfig(
        config=base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={
            "numpy": "import numpy as np",
            "time": time_prolog,
        },
        python_epilog={
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        },
    )



    return returnn_config
