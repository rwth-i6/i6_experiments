__all__ = ["run", "run_single"]

import copy
import dataclasses
import itertools

import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------
import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.nn import oclr
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.factored import PhonemeStateClasses, RasrStateTying
from ...setups.fh.network.augment import remove_label_pops_and_losses_from_returnn_config
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CART_TREE_DI,
    CART_TREE_DI_NUM_LABELS,
    CART_TREE_TRI,
    CART_TREE_TRI_NUM_LABELS,
    CONF_CHUNKING,
    CONF_FOCAL_LOSS,
    CONF_SA_CONFIG,
    GMM_TRI_ALIGNMENT,
    RASR_ARCH,
    RASR_ROOT_NO_TF,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_NO_TF, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RASR_TF_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF2")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    tri_gmm_align = tk.Path(GMM_TRI_ALIGNMENT, cached=True)

    for (n_phones, cart_tree, cart_num_labels, lr) in [
        (3, CART_TREE_TRI, CART_TREE_TRI_NUM_LABELS, "v6"),
        (2, CART_TREE_DI, CART_TREE_DI_NUM_LABELS, "v6"),
        (3, CART_TREE_TRI, CART_TREE_TRI_NUM_LABELS, "v13"),
        (2, CART_TREE_DI, CART_TREE_DI_NUM_LABELS, "v13"),
    ]:
        with open(cart_num_labels, "r") as file:
            num_labels = int(file.read().strip())

        run_single(
            alignment=tri_gmm_align,
            alignment_name="GMMtri",
            focal_loss=CONF_FOCAL_LOSS,
            returnn_root=returnn_root,
            tune_decoding=False,
            cart_tree=tk.Path(cart_tree, cached=True),
            n_cart_phones=n_phones,
            n_cart_out=num_labels,
            lr=lr,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    returnn_root: tk.Path,
    n_cart_phones: int,
    n_cart_out: int,
    cart_tree: tk.Path,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
    focal_loss: float = CONF_FOCAL_LOSS,
    dc_detection: bool = False,
    tune_decoding: bool = False,
    lr: str = "v6",
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"blstm-{n_cart_phones}-ep:{num_epochs}-lr:{lr}-fl:{focal_loss}"
    print(f"cart {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=dc_detection)
    data_preparation_args = gmm_setups.get_final_output(name="data_preparation")
    # *********** System Instantiation *****************
    steps = rasr_util.RasrSteps()
    steps.add_step("init", None)  # you can create the label_info and pass here
    s = fh_system.FactoredHybridSystem(
        rasr_binary_path=RASR_BINARY_PATH,
        rasr_init_args=rasr_init_args,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    s.train_key = train_key
    s.label_info = dataclasses.replace(
        s.label_info, state_tying=RasrStateTying.cart, phoneme_state_classes=PhonemeStateClasses.none
    )
    s.lm_gc_simple_hash = True
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    for crp_name in s.crp_names.values():
        s.crp[crp_name].acoustic_model_config.state_tying.file = cart_tree

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    blstm_size = 512
    network = {
        "source": {
            "class": "eval",
            "eval": "self.network.get_config().typed_value('transform')(source(0), network=self.network)",
            "from": "data",
        },
        "lstm_bwd_1": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["source"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_bwd_2": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_1", "lstm_bwd_1"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_bwd_3": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_2", "lstm_bwd_2"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_bwd_4": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_3", "lstm_bwd_3"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_bwd_5": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_4", "lstm_bwd_4"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_bwd_6": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_5", "lstm_bwd_5"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_fwd_1": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["source"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_fwd_2": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_1", "lstm_bwd_1"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_fwd_3": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_2", "lstm_bwd_2"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_fwd_4": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_3", "lstm_bwd_3"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_fwd_5": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_4", "lstm_bwd_4"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "lstm_fwd_6": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_5", "lstm_bwd_5"],
            "n_out": blstm_size,
            "unit": "nativelstm2",
        },
        "output": {
            "class": "softmax",
            "from": ["lstm_fwd_6", "lstm_bwd_6"],
            "loss": "ce",
            "loss_opts": {"focal_loss_factor": focal_loss},
            "target": "classes",
            "n_out": n_cart_out,
        },
    }

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "batch_size": 12500,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": "64:32",
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {
                "dim": 50,
                "shape": (None, 50),
            },
            "classes": {
                "dim": n_cart_out,
                "shape": (None,),
                "sparse": True,
            },
        },
    }
    keep_epochs = [550, num_epochs]
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

    s.set_experiment_dict("fh", alignment_name, "tri" if n_cart_phones == 3 else "di", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "returnn_config": returnn_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
    }
    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        on_2080=False,
    )
    s.set_mono_priors_returnn_rasr(
        key="fh",
        epoch=keep_epochs[-2],
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        output_layer_name="output",
        smoothen=True,
        returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
    )

    decoding_config = copy.deepcopy(returnn_config)
    decoding_config.config["network"]["output"]["class"] = "linear"
    decoding_config.config["network"]["output"]["activation"] = "log_softmax"
    for layer in decoding_config.config["network"].values():
        layer.pop("target", None)
        layer.pop("loss", None)
        layer.pop("loss_scale", None)
        layer.pop("loss_opts", None)

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        s.recognize_cart(
            key="fh",
            epoch=ep,
            crp_corpus=crp_k,
            n_cart_out=n_cart_out,
            cart_tree_or_tying_config=cart_tree,
            log_softmax_returnn_config=decoding_config,
            params=s.get_cart_params(key="fh"),
            native_ops=["NativeLstm2"],
            opt_lm_am_scale=False,
        )

    if n_cart_phones == 3:
        for cfg in [
            dataclasses.replace(s.get_cart_params(key="fh"), altas=a, beam=b)
            for a, b in itertools.product([None, 2, 4, 6], [14, 16])
        ]:
            job = s.recognize_cart(
                key="fh",
                epoch=max(keep_epochs),
                crp_corpus="dev-other",
                n_cart_out=n_cart_out,
                cart_tree_or_tying_config=cart_tree,
                log_softmax_returnn_config=decoding_config,
                calculate_statistics=True,
                params=cfg,
                opt_lm_am_scale=False,
                cpu_rqmt=2,
                mem_rqmt=4,
                rtf=4,
                native_ops=["NativeLstm2"],
            )
            job.rqmt.update({"sbatch_args": ["-w", "cn-30"]})

    return s
