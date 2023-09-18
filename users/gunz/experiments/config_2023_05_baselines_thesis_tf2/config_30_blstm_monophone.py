__all__ = ["run", "run_single"]

import copy
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.nn import oclr, returnn_time_tag
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.factored import PhoneticContext
from ...setups.fh.network import extern_data
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    BLSTM_FH_DECODING_TENSOR_CONFIG,
    CONF_CHUNKING,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
    CONF_SA_CONFIG,
    FROM_SCRATCH_CV_INFO,
    L2,
    GMM_TRI_ALIGNMENT,
    RASR_ARCH,
    RASR_ROOT_NO_TF,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
    SCRATCH_ALIGNMENT,
    SCRATCH_ALIGNMENT_DANIEL,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_NO_TF, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RASR_TF_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF2")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True)
class Experiment:
    alignment: tk.Path
    alignment_name: str
    lr: str
    dc_detection: bool
    tune_decoding: bool
    own_priors: bool

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    scratch_align = tk.Path(SCRATCH_ALIGNMENT, cached=True)
    scratch_align_daniel = tk.Path(SCRATCH_ALIGNMENT_DANIEL, cached=True)
    tri_gmm_align = tk.Path(GMM_TRI_ALIGNMENT, cached=True)

    configs = [
        Experiment(
            alignment=tri_gmm_align,
            alignment_name="GMMtri",
            dc_detection=False,
            lr="v13",
            own_priors=False,
            tune_decoding=False,
        ),
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            dc_detection=False,
            lr="v6",
            own_priors=True,
            tune_decoding=False,
        ),
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            dc_detection=False,
            lr="v13",
            own_priors=True,
            tune_decoding=False,
        ),
        # Experiment(
        #     alignment=scratch_align_daniel,
        #     alignment_name="scratch_daniel",
        #     dc_detection=True,
        #     lr="v13",
        #     own_priors=True,
        #     tune_decoding=False,
        # ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            dc_detection=exp.dc_detection,
            focal_loss=exp.focal_loss,
            returnn_root=returnn_root,
            tune_decoding=exp.tune_decoding,
            lr=exp.lr,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    returnn_root: tk.Path,
    num_epochs: int = 600,
    focal_loss: float,
    dc_detection: bool,
    tune_decoding: bool,
    lr: str,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"blstm-1-a:{alignment_name}-lr:{lr}-fl:{focal_loss}"
    print(f"fh {name}")

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
    if alignment_name == "scratch_daniel":
        s.cv_info = FROM_SCRATCH_CV_INFO
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
        is_cv_separate_from_train=alignment_name == "scratch_daniel",
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
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
        "encoder-output": {
            "class": "copy",
            "from": ["lstm_fwd_6", "lstm_bwd_6"],
            "n_out": 2 * blstm_size,
        },
    }
    network = augment_net_with_label_pops(network, label_info=s.label_info)
    network = augment_net_with_monophone_outputs(
        network,
        add_mlps=True,
        encoder_output_len=2 * blstm_size,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=True,
    )

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
                "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)},
            },
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=time_tag_name),
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
        python_prolog={"numpy": "import numpy as np", "time": time_prolog},
        python_epilog={
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        },
    )

    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)
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
        smoothen=True,
        returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
    )

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=BLSTM_FH_DECODING_TENSOR_CONFIG,
            tf_library=[s.native_lstm2_job.out_op],
        )
        for cfg in [recog_args.with_prior_scale(0.6).with_tdp_scale(0.5)]:
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=2 * blstm_size,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                rtf_cpu=30,
            )

        if tune_decoding:
            best_config = recognizer.recognize_optimize_scales(
                label_info=s.label_info,
                search_parameters=recog_args,
                num_encoder_output=2 * blstm_size,
                prior_scales=np.linspace(0.0, 0.6, 7),
                tdp_scales=np.linspace(0.2, 0.6, 5),
            )
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=best_config,
                num_encoder_output=2 * blstm_size,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                name_override="best/4gram",
                rtf_cpu=30,
            )

    return s
