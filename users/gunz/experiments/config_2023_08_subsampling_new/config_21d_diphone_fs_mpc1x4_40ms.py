__all__ = ["run", "run_single"]

import copy
import dataclasses
import math
import typing
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

from ...setups.common.nn import baum_welch, oclr, returnn_time_tag
from ...setups.common.nn.chunking import subsample_chunking
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.network import conformer, diphone_joint_output
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network.augment import (
    SubsamplingInfo,
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING_10MS,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
    CONF_SA_CONFIG,
    L2,
    RASR_ARCH,
    RASR_ROOT_NO_TF,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_NO_TF, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RASR_TF_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF2")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True)
class Experiment:
    adapt_transition_model_to_ss: bool
    alignment: tk.Path
    batch_size: int
    decode_all_corpora: bool
    fine_tune: bool
    lr: str
    dc_detection: bool
    num_epochs: int
    run_performance_study: bool
    tune_decoding: bool
    run_tdp_study: bool

    filter_segments: typing.Optional[typing.List[str]] = None
    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path, alignment: tk.Path, a_name: str):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    configs = [
        Experiment(
            adapt_transition_model_to_ss=False,
            alignment=alignment,
            batch_size=10000,
            dc_detection=False,
            decode_all_corpora=False,
            fine_tune=a_name == "40ms-FF-v8",
            lr="v13",
            num_epochs=600,
            run_performance_study=False,  # a_name == "40ms-FF-v8",
            tune_decoding=a_name == "40ms-FF-v8",
            run_tdp_study=False,
        ),
        Experiment(
            adapt_transition_model_to_ss=False,
            alignment=alignment,
            batch_size=10000,
            dc_detection=False,
            decode_all_corpora=False,
            fine_tune=a_name == "40ms-FF-v8",
            lr="v13",
            # 200 is the equivalent number of epochs in training time of training an FF-NN for alignment
            num_epochs=600 + 600,
            run_performance_study=False,  # a_name == "40ms-FF-v8",
            tune_decoding=a_name == "40ms-FF-v8",
            run_tdp_study=False,
        ),
        Experiment(
            adapt_transition_model_to_ss=True,
            alignment=alignment,
            batch_size=10000,
            dc_detection=False,
            decode_all_corpora=False,
            fine_tune=a_name == "40ms-FF-v8",
            lr="v13",
            # 200 is the equivalent number of epochs in training time of training an FF-NN for alignment
            num_epochs=600 + 600,
            run_performance_study=False,  # a_name == "40ms-FF-v8",
            tune_decoding=a_name == "40ms-FF-v8",
            run_tdp_study=False,
        ),
    ]
    for exp in configs:
        run_single(
            adapt_transition_model_to_ss=exp.adapt_transition_model_to_ss,
            alignment=exp.alignment,
            batch_size=exp.batch_size,
            dc_detection=exp.dc_detection,
            decode_all_corpora=exp.decode_all_corpora,
            fine_tune=exp.fine_tune,
            focal_loss=exp.focal_loss,
            num_epochs=exp.num_epochs,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            tune_decoding=exp.tune_decoding,
            filter_segments=exp.filter_segments,
            lr=exp.lr,
            run_tdp_study=exp.run_tdp_study,
        )


def run_single(
    *,
    adapt_transition_model_to_ss: bool,
    alignment: tk.Path,
    batch_size: int,
    dc_detection: bool,
    decode_all_corpora: bool,
    fine_tune: bool,
    focal_loss: float,
    lr: str,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    run_tdp_study: bool,
    filter_segments: typing.Optional[typing.List[str]],
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    bw_scale = baum_welch.BwScales(label_posterior_scale=0.3, label_prior_scale=None, transition_scale=0.3)
    ss_factor = 4

    name = f"conf-2-ep:{num_epochs}-lr:{lr}-fl:{focal_loss}-fs-bwl:{bw_scale.label_posterior_scale}-bwt:{bw_scale.transition_scale}-tdp:{'adapted' if adapt_transition_model_to_ss else 'classic'}"
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
    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1, state_tying=RasrStateTying.diphone)
    s.lm_gc_simple_hash = True
    s.train_key = train_key
    if filter_segments is not None:
        s.filter_segments = filter_segments
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
        chunk_size=CONF_CHUNKING_10MS,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="heuristic-40ms" if adapt_transition_model_to_ss else "heuristic",
        eval_tdp_type="heuristic-40ms" if adapt_transition_model_to_ss else "heuristic",
        add_base_allophones=False,
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=CONF_CHUNKING_10MS,
        focal_loss_factor=CONF_FOCAL_LOSS,
        label_smoothing=CONF_LABEL_SMOOTHING,
        num_classes=s.label_info.get_n_of_dense_classes(),
        time_tag_name=time_tag_name,
        upsample_by_transposed_conv=False,
        conf_args={
            "feature_stacking": False,
            "reduction_factor": (1, ss_factor),
        },
    )
    network = network_builder.network
    network = augment_net_with_monophone_outputs(
        network,
        add_mlps=True,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=True,
    )
    network = augment_net_with_diphone_outputs(
        network,
        encoder_output_len=conf_model_dim,
        label_smoothing=CONF_LABEL_SMOOTHING,
        l2=L2,
        ph_emb_size=s.label_info.ph_emb_size,
        st_emb_size=s.label_info.st_emb_size,
        use_multi_task=True,
    )

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "batch_size": batch_size,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": subsample_chunking(CONF_CHUNKING_10MS, ss_factor),
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
        },
        "dev": {"reduce_target_factor": ss_factor},
        "train": {"reduce_target_factor": ss_factor},
    }
    keep_epochs = [100, 300, 400, 500, 550, 600]
    if num_epochs > 600:
        factor = num_epochs // 600
        multiplied = np.array(keep_epochs) * factor
        keep_epochs = [int(v) for v in multiplied]
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
    returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    nn_precomputed_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=True,
    )
    prior_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=False,
    )
    returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=True,
        prepare_for_train=True,
    )
    returnn_config = baum_welch.augment_for_fast_bw(
        crp=s.crp[s.crp_names["train"]],
        from_output_layer="output",
        returnn_config=returnn_config,
        log_linear_scales=bw_scale,
    )

    s.set_experiment_dict("fh", "fs", "di", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "returnn_config": copy.deepcopy(returnn_config),
    }
    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    for ep, crp_k in itertools.product(keep_epochs, ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=min(ep, keep_epochs[-2]),
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(prior_config, except_layers=["pastLabel"]),
            output_layer_name="output",
        )

        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "diphone-dense"

        base_params = s.get_cart_params(key="fh")
        decoding_cfgs = [
            dataclasses.replace(
                base_params,
                lm_scale=base_params.lm_scale / ss_factor,
                tdp_speech=(10, 0, "infinity", 0),
                tdp_silence=(10, 10, "infinity", 10),
                tdp_scale=sc,
            ).with_prior_scale(pC)
            for sc, pC in [(0.4, 0.3), (0.2, 0.4), (0.4, 0.4), (0.2, 0.5)]
        ]
        for cfg in decoding_cfgs:
            s.recognize_cart(
                key="fh",
                epoch=ep,
                crp_corpus=crp_k,
                n_cart_out=s.label_info.get_n_of_dense_classes(),
                cart_tree_or_tying_config=tying_cfg,
                params=cfg,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                calculate_statistics=True,
                opt_lm_am_scale=True,
                prior_epoch=min(ep, keep_epochs[-2]),
                rtf=12,
            )

    return s
