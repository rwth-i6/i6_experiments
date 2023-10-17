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
from ...setups.fh.decoder.config import SearchParameters
from ...setups.fh.network import conformer, diphone_joint_output
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    DEFAULT_INIT,
    SubsamplingInfo,
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING_10MS,
    CONF_FH_DECODING_TENSOR_CONFIG,
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
    alignment: tk.Path
    alignment_name: str
    batch_size: int
    chunking: str
    decode_all_corpora: bool
    fine_tune: bool
    global_l2: bool
    init: str
    label_smoothing: float
    lr: str
    dc_detection: bool
    run_performance_study: bool
    tune_decoding: bool
    run_tdp_study: bool
    sa_max_reps_t: int
    ss_strategy: str
    smooth_oclr: bool
    swap_mhsa_conv: bool

    filter_segments: typing.Optional[typing.List[str]] = None
    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path, alignment: tk.Path, a_name: str):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    frontend_configs = [
        Experiment(
            alignment=alignment,
            alignment_name=a_name,
            batch_size=12500,
            chunking=CONF_CHUNKING_10MS,
            dc_detection=False,
            decode_all_corpora=False,
            fine_tune=False,
            global_l2=False,
            init=DEFAULT_INIT,
            label_smoothing=CONF_LABEL_SMOOTHING,
            lr="v13",
            run_performance_study=a_name == "40ms-FF-v8",
            tune_decoding=a_name == "40ms-FF-v8",
            run_tdp_study=False,
            smooth_oclr=False,
            sa_max_reps_t=20,
            ss_strategy=ss,
            swap_mhsa_conv=False,
        )
        for ss in ["mp:2@2+mp:2@4", "mp:4@2", "cs:2@2+cs:2@2", "cs:4@2", "cs:4@4"]
    ]
    encoder_configs = [
        Experiment(
            alignment=alignment,
            alignment_name=a_name,
            batch_size=12500,
            chunking=CONF_CHUNKING_10MS,
            dc_detection=False,
            decode_all_corpora=False,
            fine_tune=True,
            global_l2=global_l2,
            init=w_init,
            label_smoothing=CONF_LABEL_SMOOTHING,
            lr="v13",
            run_performance_study=False,
            tune_decoding=a_name == "40ms-FFs-v8",
            run_tdp_study=False,
            sa_max_reps_t=20,
            ss_strategy="mp:4@4",
            smooth_oclr=False,
            swap_mhsa_conv=swap_mhsa_conv,
        )
        for w_init, swap_mhsa_conv, global_l2 in itertools.product(
            [
                "glorot_uniform",  # RETURNN default
                "he_normal",  # Glorot adapted for ReLU
            ],
            [False, True],
            [False, True],
        )
        if a_name == "40ms-FFs-v8"
    ]
    smooth_lr_configs = (
        [
            Experiment(
                alignment=alignment,
                alignment_name=a_name,
                batch_size=12500,
                chunking=CONF_CHUNKING_10MS,
                dc_detection=False,
                decode_all_corpora=False,
                fine_tune=True,
                global_l2=False,
                init=DEFAULT_INIT,
                label_smoothing=CONF_LABEL_SMOOTHING,
                lr="v13",
                run_performance_study=False,
                tune_decoding=a_name == "40ms-FFs-v8",
                run_tdp_study=False,
                sa_max_reps_t=20,
                ss_strategy="mp:4@4",
                smooth_oclr=True,
                swap_mhsa_conv=False,
            )
        ]
        if a_name == "40ms-FFs-v8"
        else []
    )
    sa_configs = [
        Experiment(
            alignment=alignment,
            alignment_name=a_name,
            batch_size=12500,
            chunking=CONF_CHUNKING_10MS,
            dc_detection=False,
            decode_all_corpora=False,
            fine_tune=True,
            global_l2=False,
            init=DEFAULT_INIT,
            label_smoothing=CONF_LABEL_SMOOTHING,
            lr="v13",
            run_performance_study=False,
            tune_decoding=a_name == "40ms-FFs-v8",
            run_tdp_study=False,
            sa_max_reps_t=sa,
            ss_strategy="mp:4@4",
            smooth_oclr=False,
            swap_mhsa_conv=False,
        )
        for sa in [4, 10, 14, 20, 25]
        if a_name == "40ms-FFs-v8"
    ]
    for exp in [*frontend_configs, *encoder_configs, *smooth_lr_configs, *sa_configs]:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            batch_size=exp.batch_size,
            chunking=exp.chunking,
            dc_detection=exp.dc_detection,
            decode_all_corpora=exp.decode_all_corpora,
            fine_tune=exp.fine_tune,
            focal_loss=exp.focal_loss,
            global_l2=exp.global_l2,
            label_smoothing=exp.label_smoothing,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            tune_decoding=exp.tune_decoding,
            filter_segments=exp.filter_segments,
            lr=exp.lr,
            run_tdp_study=exp.run_tdp_study,
            smooth_oclr=exp.smooth_oclr,
            sa_max_reps_t=exp.sa_max_reps_t,
            ss_strategy=exp.ss_strategy,
            swap_mhsa_conv=exp.swap_mhsa_conv,
            weights_init=exp.init,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    batch_size: int,
    chunking: str,
    dc_detection: bool,
    decode_all_corpora: bool,
    fine_tune: bool,
    focal_loss: float,
    global_l2: bool,
    lr: str,
    returnn_root: tk.Path,
    run_performance_study: bool,
    ss_strategy: str,
    tune_decoding: bool,
    run_tdp_study: bool,
    label_smoothing: float = CONF_LABEL_SMOOTHING,
    filter_segments: typing.Optional[typing.List[str]],
    weights_init: str,
    swap_mhsa_conv: bool,
    smooth_oclr: bool,
    sa_max_reps_t: int,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = (
        f"conf-2-{ss_strategy}-a:{alignment_name}-lr:{lr}-fl:{focal_loss}-ls:{label_smoothing}-ch:{chunking}-sa_t"
        f":{sa_max_reps_t}"
    )
    if weights_init != DEFAULT_INIT:
        name += f"-init:{weights_init}"
    else:
        name += "-init:default"
    if global_l2:
        name += "-l2"
    if swap_mhsa_conv:
        name += "-swap"
    if smooth_oclr:
        name += "-smooth"
    print(f"fh {name}")

    ss_factor = 4

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
    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1)
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
        chunk_size=chunking,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

    conv_args = {
        "cs:4@2": {
            "conv0_1": {
                "strides": (4, 1),
            },
        },
        "cs:4@4": {
            "conv1_1": {
                "strides": (4, 1),
            },
        },
        "cs:2@2+cs:2@2": {
            "conv0_1": {
                "strides": (2, 1),
            },
            "conv1_1": {
                "strides": (2, 1),
            },
        },
    }
    pooling_size = {
        "mp:2@2+mp:2@4": (2, 2),
        "mp:4@2": (4, 1),
        "mp:4@4": (1, 4),
    }
    reduction_pattern = {
        "cs:2@2+cs:2@2": [2, 2],
        "mp:2@2+mp:2@4": [2, 2],
        "cs:4@2": 4,
        "cs:4@4": 4,
        "mp:4@2": 4,
        "mp:4@4": 4,
    }

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=chunking,
        focal_loss_factor=CONF_FOCAL_LOSS,
        label_smoothing=label_smoothing,
        num_classes=s.label_info.get_n_of_dense_classes(),
        time_tag_name=time_tag_name,
        upsample_by_transposed_conv=False,
        weights_init=weights_init,
        conf_args={
            "feature_stacking": False,
            "conv_args": conv_args.get(ss_strategy, None),
            "reduction_factor": pooling_size.get(ss_strategy, None),
        },
    )
    network = network_builder.network
    network = augment_net_with_label_pops(
        network,
        label_info=s.label_info,
        classes_subsampling_info=SubsamplingInfo(factor=reduction_pattern[ss_strategy], time_tag_name=time_tag_name),
    )
    network = augment_net_with_monophone_outputs(
        network,
        add_mlps=True,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=label_smoothing,
        use_multi_task=True,
        weights_init=weights_init,
    )
    network = augment_net_with_diphone_outputs(
        network,
        encoder_output_len=conf_model_dim,
        label_smoothing=label_smoothing,
        l2=L2,
        ph_emb_size=s.label_info.ph_emb_size,
        st_emb_size=s.label_info.st_emb_size,
        use_multi_task=True,
    )
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=label_smoothing,
        time_tag_name=time_tag_name,
        upsampling=False,
    )

    if global_l2:
        for layer in network.values():
            if layer.get("class", "").lower() in ["conv", "linear", "softmax"]:
                layer["L2"] = L2

    if swap_mhsa_conv:
        for i in range(1, 12 + 1):
            network[f"enc_{i:03d}_conv_laynorm"]["from"] = [f"enc_{i:03d}_ff1_out"]
            network[f"enc_{i:03d}_conv_output"]["from"] = [f"enc_{i:03d}_ff1_out", f"enc_{i:03d}_conv_dropout"]

            network[f"enc_{i:03d}_self_att_laynorm"]["from"] = [f"enc_{i:03d}_conv_output"]
            network[f"enc_{i:03d}_self_att_out"]["from"] = [f"enc_{i:03d}_conv_output", f"enc_{i:03d}_self_att_drop"]

            network[f"enc_{i:03d}_ff2_laynorm"]["from"] = [f"enc_{i:03d}_self_att_out"]
            network[f"enc_{i:03d}_ff2_out"]["from"] = [f"enc_{i:03d}_self_att_out", f"enc_{i:03d}_ff2_drop_half"]

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "max_reps_time": sa_max_reps_t,
        "batch_size": batch_size,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": subsample_chunking(chunking, ss_factor),
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=None),
        },
        "dev": {"reduce_target_factor": ss_factor},
        "train": {"reduce_target_factor": ss_factor},
    }
    keep_epochs = [100, 300, 400, 500, 550, num_epochs]
    base_post_config = {
        "cleanup_old_models": {
            "keep_best_n": 3,
            "keep": keep_epochs,
        },
    }
    epilog = {
        "functions": [
            sa_mask,
            sa_random_mask,
            sa_summary,
            sa_transform,
        ],
    }
    if smooth_oclr:
        epilog["lr"] = dynamic_learning_rate
    returnn_config = returnn.ReturnnConfig(
        config=base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={
            "numpy": "import numpy as np",
            "time": time_prolog,
        },
        python_epilog=epilog,
    )

    s.set_experiment_dict("fh", alignment_name, "di", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "returnn_config": copy.deepcopy(returnn_config),
    }
    viterbi_train_j = s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    clean_returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    nn_precomputed_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=clean_returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=True,
    )
    prior_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
        returnn_config=clean_returnn_config,
        label_info=s.label_info,
        out_joint_score_layer="output",
        log_softmax=False,
    )

    for ep, crp_k in itertools.product(keep_epochs, ["dev-other"]):
        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=min(ep, keep_epochs[-2]),
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=prior_returnn_config,
            output_layer_name="output",
        )

        diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "diphone-dense"

        configs = [
            dataclasses.replace(
                s.get_cart_params("fh"), beam=beam, beam_limit=100000, lm_scale=2, tdp_scale=tdpS
            ).with_prior_scale(pC)
            for beam, pC, tdpS in itertools.product(
                [18, 20] if ep == max(keep_epochs) else [18],
                [0.4, 0.6],
                [0.4, 0.6],
            )
        ]
        for cfg in configs:
            s.recognize_cart(
                key="fh",
                epoch=ep,
                calculate_statistics=True,
                cart_tree_or_tying_config=tying_cfg,
                cpu_rqmt=2,
                opt_lm_am_scale=ep == max(keep_epochs),
                crp_corpus=crp_k,
                lm_gc_simple_hash=True,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                mem_rqmt=4,
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                params=cfg,
                rtf=4,
            )

    # ###########
    # FINE TUNING
    # ###########

    if fine_tune:
        fine_tune_epochs = 450
        keep_epochs = [23, 100, 225, 400, 450]
        orig_name = name

        bw_scales = [baum_welch.BwScales(label_posterior_scale=1.0, label_prior_scale=None, transition_scale=0.3)]
        configs = [(8e-5, scales) for scales in bw_scales]

        for peak_lr, bw_scale in configs:
            name = f"{orig_name}-fs:{peak_lr}-bwl:{bw_scale.label_posterior_scale}-bwt:{bw_scale.transition_scale}"
            s.set_experiment_dict("fh-fs", alignment_name, "di", postfix_name=name)

            s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
            s.lexicon_args["norm_pronunciation"] = False

            s._update_am_setting_for_all_crps(
                train_tdp_type="heuristic-40ms",
                eval_tdp_type="heuristic-40ms",
                add_base_allophones=False,
            )

            returnn_config_ft = remove_label_pops_and_losses_from_returnn_config(returnn_config)
            nn_precomputed_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
                returnn_config=returnn_config_ft,
                label_info=s.label_info,
                out_joint_score_layer="output",
                log_softmax=True,
            )
            prior_config = diphone_joint_output.augment_to_joint_diphone_softmax(
                returnn_config=returnn_config_ft,
                label_info=s.label_info,
                out_joint_score_layer="output",
                log_softmax=False,
            )
            returnn_config_ft = diphone_joint_output.augment_to_joint_diphone_softmax(
                returnn_config=returnn_config_ft,
                label_info=s.label_info,
                out_joint_score_layer="output",
                log_softmax=True,
                prepare_for_train=True,
            )
            returnn_config_ft = baum_welch.augment_for_fast_bw(
                crp=s.crp[s.crp_names["train"]],
                from_output_layer="output",
                returnn_config=returnn_config_ft,
                log_linear_scales=bw_scale,
            )
            lrates = oclr.get_learning_rates(
                lrate=peak_lr,
                increase=0,
                constLR=math.floor(fine_tune_epochs * 0.45),
                decay=math.floor(fine_tune_epochs * 0.45),
                decMinRatio=0.1,
                decMaxRatio=1,
            )
            update_config = returnn.ReturnnConfig(
                config={
                    "batch_size": 10000,
                    "learning_rates": list(
                        np.concatenate([lrates, np.linspace(min(lrates), 1e-6, fine_tune_epochs - len(lrates))])
                    ),
                    "preload_from_files": {
                        "existing-model": {
                            "init_for_train": True,
                            "ignore_missing": True,
                            "filename": viterbi_train_j.out_checkpoints[600],
                        }
                    },
                    "extern_data": {"data": {"dim": 50}},
                },
                post_config={"cleanup_old_models": {"keep_best_n": 3, "keep": keep_epochs}},
                python_epilog={
                    "dynamic_lr_reset": "dynamic_learning_rate = None",
                },
            )
            returnn_config_ft.update(update_config)

            s.set_returnn_config_for_experiment("fh-fs", copy.deepcopy(returnn_config_ft))

            train_args = {
                **s.initial_train_args,
                "num_epochs": fine_tune_epochs,
                "partition_epochs": partition_epochs,
                "returnn_config": copy.deepcopy(returnn_config_ft),
            }
            s.returnn_rasr_training(
                experiment_key="fh-fs",
                train_corpus_key=s.crp_names["train"],
                dev_corpus_key=s.crp_names["cvtrain"],
                nn_train_args=train_args,
            )

            for ep, crp_k in itertools.product(keep_epochs, ["dev-other"]):
                s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

                s.set_mono_priors_returnn_rasr(
                    key="fh-fs",
                    epoch=min(ep, keep_epochs[-2]),
                    train_corpus_key=s.crp_names["train"],
                    dev_corpus_key=s.crp_names["cvtrain"],
                    smoothen=True,
                    returnn_config=remove_label_pops_and_losses_from_returnn_config(
                        prior_config, except_layers=["pastLabel"]
                    ),
                    output_layer_name="output",
                )

                diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
                tying_cfg = rasr.RasrConfig()
                tying_cfg.type = "diphone-dense"

                base_params = s.get_cart_params(key="fh-fs")
                decoding_cfgs = [
                    dataclasses.replace(
                        base_params,
                        lm_scale=round(base_params.lm_scale / ss_factor, 2),
                        tdp_scale=sc,
                    ).with_prior_scale(0.6)
                    for sc in [0.4, 0.6]
                ]
                for cfg in decoding_cfgs:
                    s.recognize_cart(
                        key="fh-fs",
                        epoch=ep,
                        crp_corpus=crp_k,
                        n_cart_out=diphone_li.get_n_of_dense_classes(),
                        cart_tree_or_tying_config=tying_cfg,
                        params=cfg,
                        log_softmax_returnn_config=nn_precomputed_returnn_config,
                        calculate_statistics=True,
                        opt_lm_am_scale=True,
                        prior_epoch=min(ep, keep_epochs[-2]),
                        rtf=8,
                        cpu_rqmt=2,
                        mem_rqmt=4,
                    )

                if run_performance_study:
                    configs = [
                        dataclasses.replace(
                            s.get_cart_params("fh"), altas=a, beam=beam, beam_limit=100000, lm_scale=2, tdp_scale=0.4
                        ).with_prior_scale(pC)
                        for beam, pC, a in itertools.product(
                            [14, 16, 18],
                            [0.4, 0.6],
                            [None, 2, 4, 6],
                        )
                    ]
                    for cfg in configs:
                        j = s.recognize_cart(
                            key="fh-fs",
                            epoch=23,
                            calculate_statistics=True,
                            cart_tree_or_tying_config=tying_cfg,
                            cpu_rqmt=2,
                            crp_corpus="dev-other",
                            lm_gc_simple_hash=True,
                            log_softmax_returnn_config=nn_precomputed_returnn_config,
                            mem_rqmt=4,
                            n_cart_out=diphone_li.get_n_of_dense_classes(),
                            params=cfg,
                            prior_epoch=min(ep, keep_epochs[-2]),
                            rtf=1.5,
                        )
                        j.rqmt.update({"sbatch_args": ["-w", "cn-30"]})

    if run_tdp_study:
        s.feature_flows["dev-other"].flags["cache_mode"] = "bundle"
        li = dataclasses.replace(s.label_info, n_states_per_phone=1, state_tying=RasrStateTying.diphone)

        base_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
        prior_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=base_config, label_info=li, out_joint_score_layer="output", log_softmax=False
        )
        s.set_mono_priors_returnn_rasr(
            "fh",
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            epoch=max(keep_epochs),
            returnn_config=prior_returnn_config,
            output_layer_name="output",
            smoothen=True,
        )

        nn_precomputed_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=base_config, label_info=li, out_joint_score_layer="output", log_softmax=True
        )
        s.set_graph_for_experiment("fh", override_cfg=nn_precomputed_returnn_config)

        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "diphone-dense"

        search_cfg = SearchParameters.default_diphone(priors=s.experiments["fh"]["priors"]).with_prior_scale(0.5)
        tdps = itertools.product(
            [0, 3, 10],
            [0],
            [0, 3, 10],
            [0, 3, 10],
            [3],
            [1, 3, 10],
            (0.1, *((round(v, 1) for v in np.linspace(0.2, 0.8, 4)))),
        )
        for cfg in tdps:
            sp_loop, sp_fwd, sp_exit, sil_loop, sil_fwd, sil_exit, tdp_scale = cfg
            sp_tdp = (sp_loop, sp_fwd, "infinity", sp_exit)
            sil_tdp = (sil_loop, sil_fwd, "infinity", sil_exit)
            params = dataclasses.replace(
                search_cfg,
                altas=2,
                lm_scale=1.95,
                tdp_speech=sp_tdp,
                tdp_silence=sil_tdp,
                tdp_non_word=sil_tdp,
                tdp_scale=tdp_scale,
            )

            def set_concurrency(crp):
                crp.concurrent = 1

            s.recognize_cart(
                key="fh",
                crp_corpus="dev-other",
                epoch=max(keep_epochs),
                params=params,
                cart_tree_or_tying_config=tying_cfg,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                n_cart_out=li.get_n_of_dense_classes(),
                crp_update=set_concurrency,
                calculate_statistics=False,
                lm_gc_simple_hash=True,
                opt_lm_am_scale=False,
                prior_epoch=max(keep_epochs),
                mem_rqmt=2,
                cpu_rqmt=2,
                rtf=4,
            )

    if decode_all_corpora:
        assert False, "this is broken r/n"

        for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-clean", "dev-other", "test-clean", "test-other"]):
            s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

            recognizer, recog_args = s.get_recognizer_and_args(
                key="fh",
                context_type=PhoneticContext.diphone,
                crp_corpus=crp_k,
                epoch=ep,
                gpu=False,
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
                lm_gc_simple_hash=True,
            )

            cfgs = [recog_args.with_prior_scale(0.4, 0.4).with_tdp_scale(0.4)]

            for cfg in cfgs:
                recognizer.recognize_count_lm(
                    label_info=s.label_info,
                    search_parameters=cfg,
                    num_encoder_output=conf_model_dim,
                    rerun_after_opt_lm=False,
                    calculate_stats=True,
                )

            generic_lstm_base_op = returnn.CompileNativeOpJob(
                "LstmGenericBase",
                returnn_root=returnn_root,
                returnn_python_exe=RETURNN_PYTHON_EXE,
            )
            generic_lstm_base_op.rqmt = {"cpu": 1, "mem": 4, "time": 0.5}
            recognizer, recog_args = s.get_recognizer_and_args(
                key="fh",
                context_type=PhoneticContext.diphone,
                crp_corpus=crp_k,
                epoch=ep,
                gpu=True,
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
                tf_library=[generic_lstm_base_op.out_op, generic_lstm_base_op.out_grad_op],
            )

            for cfg in cfgs:
                recognizer.recognize_ls_trafo_lm(
                    label_info=s.label_info,
                    search_parameters=cfg.with_lm_scale(cfg.lm_scale + 2.0),
                    num_encoder_output=conf_model_dim,
                    rerun_after_opt_lm=False,
                    calculate_stats=True,
                    rtf_gpu=20,
                    gpu=True,
                )

    return s


# one cycle LR: triangular linear w.r.t. iterations(steps)
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    # -- need to be adjusted w.r.t. training -- #
    initialLR = 3.69e-05  # v13 initial LR
    peakLR = 0.001  # v13 peak LR
    finalLR = 1e-6
    totalEpoch = 600
    cycleEpoch = int(totalEpoch * 0.45)
    nStep = 1488  # steps/epoch depending on batch_size, manually tuned by averaging from previous training

    # -- derived -- #
    steps = cycleEpoch * nStep
    stepSize = (peakLR - initialLR) / steps
    steps2 = (totalEpoch - 2 * cycleEpoch) * nStep
    stepSize2 = (initialLR - finalLR) / steps2

    import tensorflow as tf

    n = tf.cast(global_train_step, tf.float32)
    return tf.where(
        global_train_step <= steps,
        initialLR + stepSize * n,
        tf.where(
            global_train_step <= 2 * steps,
            peakLR - stepSize * (n - steps),
            tf.maximum(initialLR - stepSize2 * (n - 2 * steps), finalLR),
        ),
    )
