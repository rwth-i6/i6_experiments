__all__ = ["run", "run_single"]

import copy
import dataclasses
import math
import typing
from typing import Dict, Any
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
from ...setups.fh import system as fh_system
from ...setups.fh.factored import LabelInfo, PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, diphone_joint_output, extern_data
from ...setups.fh.network.augment import (
    SubsamplingInfo,
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    augment_net_with_triphone_outputs,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups
from ..config_2023_05_baselines_thesis_tf2.config import SCRATCH_ALIGNMENT

from .config import (
    CONF_FH_DECODING_TENSOR_CONFIG,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
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
    decode_all_corpora: bool
    run_performance_study: bool
    tune_decoding: bool

    filter_segments: typing.Optional[typing.List[str]] = None


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    exp = Experiment(
        alignment=tk.Path(SCRATCH_ALIGNMENT, cached=True),
        alignment_name="10ms-B",
        decode_all_corpora=False,
        run_performance_study=False,
        tune_decoding=True,
    )
    sys = run_single(returnn_root=returnn_root, exp=exp)

    return exp, sys


def run_single(returnn_root: tk.Path, exp: Experiment) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()
    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=False)
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
    if exp.filter_segments is not None:
        s.filter_segments = exp.filter_segments
    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = exp.alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=False,
        input_key="data_preparation",
        chunk_size="256:64",
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
    )

    # ---------------------- returnn config---------------
    CONF_MODEL_DIM = 512
    SS_FACTOR = 4
    TENSOR_CONFIG = dataclasses.replace(
        CONF_FH_DECODING_TENSOR_CONFIG,
        in_encoder_output="conformer_12_output/add",
    )
    ZHOU_L2 = 5e-6

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    returnn_config = get_conformer_config(
        conf_model_dim=CONF_MODEL_DIM, label_info=s.label_info, time_tag_name=time_tag_name, ss_factor=SS_FACTOR
    )
    keep_epochs = [100, 200, 300, 350, 400]
    base_post_config = {
        "cleanup_old_models": {
            "keep_best_n": 3,
            "keep_last_n": 5,
            "keep": keep_epochs,
        },
    }
    update_config = returnn.ReturnnConfig(
        config=s.initial_nn_args,
        post_config=base_post_config,
        python_prolog={
            "recursion": ["import sys", "sys.setrecursionlimit(4000)"],
            "numpy": "import numpy as np",
            "time": time_prolog,
        },
    )
    returnn_config.update(update_config)

    returnn_cfg_mo = get_monophone_network(
        returnn_config=returnn_config, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    )
    returnn_cfg_di = get_diphone_network(
        returnn_config=returnn_config, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    )
    returnn_cfg_tri = get_triphone_network(
        returnn_config=returnn_config, conf_model_dim=CONF_MODEL_DIM, l2=ZHOU_L2, label_info=s.label_info
    )
    configs = [returnn_cfg_mo, returnn_cfg_di, returnn_cfg_tri]
    names = ["mono", "di", "tri"]
    keys = [f"fh-{name}" for name in names]

    for cfg, name, key in zip(configs, names, keys):
        post_name = f"conf-{name}-zhou"
        print(f"fh {post_name}")

        s.set_experiment_dict(key, exp.alignment_name, name, postfix_name=post_name)
        s.set_returnn_config_for_experiment(key, copy.deepcopy(cfg))

        train_args = {
            **s.initial_train_args,
            "num_epochs": keep_epochs[-1],
            "partition_epochs": {"train": 20, "dev": 1},
            "returnn_config": copy.deepcopy(cfg),
        }
        s.returnn_rasr_training(
            experiment_key=key,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            nn_train_args=train_args,
            on_2080=True,
        )

    for (key, returnn_config), ep, crp_k in itertools.product(zip(keys, configs), keep_epochs, ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        continue

        clean_returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
        prior_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=clean_returnn_config,
            label_info=s.label_info,
            out_joint_score_layer="output",
            log_softmax=False,
        )
        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=min(ep, keep_epochs[-2]),
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=prior_returnn_config,
        )

    # ###########
    # FINE TUNING
    # ###########

    if False:
        fine_tune_epochs = 300
        keep_epochs = [25, 150, 275, 300]
        orig_name = name

        bw_scales = [
            baum_welch.BwScales(label_posterior_scale=p, label_prior_scale=None, transition_scale=t)
            for p, t in itertools.product([0.3, 1.0], [0.0, 0.3])
        ]

        for bw_scale in bw_scales:
            name = f"{orig_name}-fs-bwl:{bw_scale.label_posterior_scale}-bwt:{bw_scale.transition_scale}"
            s.set_experiment_dict("fh-fs", alignment_name, "di", postfix_name=name)

            s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
            s.lexicon_args["norm_pronunciation"] = False

            s._update_am_setting_for_all_crps(
                train_tdp_type="heuristic",
                eval_tdp_type="heuristic",
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
                lrate=5e-5,
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
                            "filename": viterbi_train_j.out_checkpoints[400],
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
                        lm_scale=base_params.lm_scale / SS_FACTOR,
                        tdp_speech=tdp_sp,
                        tdp_silence=tdp_sil,
                        tdp_scale=sc,
                    ).with_prior_scale(pC)
                    for sc, pC in [(0.4, 0.3), (0.2, 0.4), (0.4, 0.4), (0.2, 0.5)]
                    for tdp_sp, tdp_sil in [
                        ((10, 0, "infinity", 0), (10, 10, "infinity", 10)),
                        ((3, 0, "infinity", 0), (0, 3, "infinity", 20)),
                    ]
                ]
                for cfg in decoding_cfgs:
                    trafo = (
                        False
                        and ep == max(keep_epochs)
                        and bw_scale.label_posterior_scale == 1.0
                        and bw_scale.transition_scale == 0.3
                    )
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
                        decode_trafo_lm=trafo,
                        rtf=12 * (2 if trafo else 1),
                        cpu_rqmt=2,
                    )

                if run_performance_study:
                    configs = [
                        dataclasses.replace(
                            base_params,
                            altas=a,
                            beam=beam,
                            beam_limit=100000,
                            lm_scale=2,
                            tdp_scale=0.2,
                            tdp_speech=(10, 0, "infinity", 0),
                            tdp_silence=(10, 10, "infinity", 10),
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
                            epoch=max(keep_epochs),
                            calculate_statistics=True,
                            cart_tree_or_tying_config=tying_cfg,
                            cpu_rqmt=2,
                            crp_corpus="dev-other",
                            lm_gc_simple_hash=True,
                            log_softmax_returnn_config=nn_precomputed_returnn_config,
                            mem_rqmt=4,
                            n_cart_out=diphone_li.get_n_of_dense_classes(),
                            prior_epoch=min(ep, keep_epochs[-2]),
                            params=cfg,
                            rtf=1.5,
                        )
                        j.rqmt.update({"sbatch_args": ["-p", "rescale_amd"]})

            for ep, crp_k in itertools.product([max(keep_epochs)], ["test-other"]):
                s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

                diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
                tying_cfg = rasr.RasrConfig()
                tying_cfg.type = "diphone-dense"

                base_params = s.get_cart_params(key="fh-fs")
                decoding_cfgs = [
                    dataclasses.replace(
                        base_params,
                        lm_scale=2.4,
                        tdp_speech=tdp_sp,
                        tdp_silence=tdp_sil,
                        tdp_scale=sc,
                    ).with_prior_scale(pC)
                    for sc, pC in [(0.4, 0.3), (0.2, 0.4), (0.4, 0.4), (0.2, 0.5)]
                    for tdp_sp, tdp_sil in [
                        ((10, 0, "infinity", 0), (10, 10, "infinity", 10)),
                        ((3, 0, "infinity", 0), (0, 3, "infinity", 20)),
                    ]
                ]
                for cfg in decoding_cfgs:
                    trafo = (
                        ep == max(keep_epochs)
                        and bw_scale.label_posterior_scale == 1.0
                        and bw_scale.transition_scale == 0.3
                    )
                    s.recognize_cart(
                        key="fh-fs",
                        epoch=ep,
                        crp_corpus=crp_k,
                        n_cart_out=diphone_li.get_n_of_dense_classes(),
                        cart_tree_or_tying_config=tying_cfg,
                        params=cfg,
                        log_softmax_returnn_config=nn_precomputed_returnn_config,
                        calculate_statistics=True,
                        opt_lm_am_scale=False,
                        prior_epoch=min(ep, keep_epochs[-2]),
                        decode_trafo_lm=trafo,
                        rtf=12 * (2 if trafo else 1),
                        cpu_rqmt=2,
                    )

    if False:
        assert False, "this is broken r/n"

        for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-clean", "dev-other", "test-clean", "test-other"]):
            s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

            recognizer, recog_args = s.get_recognizer_and_args(
                key="fh",
                context_type=PhoneticContext.diphone,
                crp_corpus=crp_k,
                epoch=ep,
                gpu=False,
                tensor_map=tensor_config,
                set_batch_major_for_feature_scorer=True,
            )

            cfgs = [
                cfg
                for cfg in [
                    recog_args,
                    recog_args.with_prior_scale(0.4, 0.4, 0.2).with_tdp_scale(0.6),
                    # best_config,
                ]
                if cfg is not None
            ]

            for cfg in cfgs:
                recognizer.recognize_count_lm(
                    label_info=s.label_info,
                    search_parameters=cfg,
                    num_encoder_output=CONF_MODEL_DIM,
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
                tensor_map=tensor_config,
                set_batch_major_for_feature_scorer=True,
                tf_library=[generic_lstm_base_op.out_op, generic_lstm_base_op.out_grad_op],
            )

            for cfg in cfgs:
                recognizer.recognize_ls_trafo_lm(
                    label_info=s.label_info,
                    search_parameters=cfg.with_lm_scale(cfg.lm_scale + 2.0),
                    num_encoder_output=CONF_MODEL_DIM,
                    rerun_after_opt_lm=False,
                    calculate_stats=True,
                    rtf_gpu=24,
                    gpu=True,
                )

    return s


def get_monophone_network(
    returnn_config: returnn.ReturnnConfig,
    conf_model_dim: int,
    l2: float,
    label_info: LabelInfo,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    network = augment_net_with_monophone_outputs(
        returnn_config.config["network"],
        add_mlps=True,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=l2,
        label_info=label_info,
        label_smoothing=0.1,
        use_multi_task=True,
    )
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["network"] = network
    return returnn_config


def get_diphone_network(
    returnn_config: returnn.ReturnnConfig,
    conf_model_dim: int,
    l2: float,
    label_info: LabelInfo,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    network = augment_net_with_monophone_outputs(
        returnn_config.config["network"],
        add_mlps=True,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=l2,
        label_info=label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=True,
    )
    network = augment_net_with_diphone_outputs(
        network,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        l2=l2,
        label_smoothing=CONF_LABEL_SMOOTHING,
        ph_emb_size=label_info.ph_emb_size,
        st_emb_size=label_info.st_emb_size,
        use_multi_task=True,
    )
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["network"] = network
    return returnn_config


def get_triphone_network(
    returnn_config: returnn.ReturnnConfig,
    conf_model_dim: int,
    l2: float,
    label_info: LabelInfo,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    network = augment_net_with_monophone_outputs(
        returnn_config.config["network"],
        add_mlps=True,
        encoder_output_layer=out_layer_name,
        encoder_output_len=conf_model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=l2,
        label_info=label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        use_multi_task=True,
    )
    network = augment_net_with_triphone_outputs(
        network,
        encoder_output_layer=out_layer_name,
        l2=l2,
        ph_emb_size=label_info.ph_emb_size,
        st_emb_size=label_info.st_emb_size,
        variant=PhoneticContext.triphone_forward,
    )
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["network"] = network
    return returnn_config


def get_conformer_config(
    conf_model_dim: int,
    label_info: LabelInfo,
    time_tag_name: str,
    ss_factor: int = 4,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    assert ss_factor == 4, "unimplemented"

    ZHOU_L2 = 5e-6
    network = {
        "input_dropout": {"class": "copy", "dropout": 0.1, "from": "input_linear"},
        "input_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conv_merged",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "source": {
            "class": "eval",
            "from": "data",
            "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
        },
        "conv_1": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_source",
            "n_out": 32,
            "padding": "same",
            "with_bias": True,
        },
        "conv_1_pool": {
            "class": "pool",
            "from": "conv_1",
            "mode": "max",
            "padding": "same",
            "pool_size": (1, 2),
            "trainable": False,
        },
        "conv_2": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_1_pool",
            "n_out": 64,
            "padding": "same",
            "strides": (2, 1),
            "with_bias": True,
        },
        "conv_3": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_2",
            "n_out": 64,
            "padding": "same",
            "strides": (2, 1),
            "with_bias": True,
        },
        "conv_merged": {"axes": "static", "class": "merge_dims", "from": "conv_3"},
        "conv_source": {"axis": "F", "class": "split_dims", "dims": (-1, 1), "from": "source"},
        "conformer_01_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_01_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_01_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_01_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_01_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_conv_mod_pointwise_conv_2",
        },
        "conformer_01_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_01_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_01_conv_mod_ln": {"class": "layer_norm", "from": "conformer_01_ffmod_1_half_res_add"},
        "conformer_01_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_01_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_01_conv_mod_dropout", "conformer_01_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_01_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_01_conv_mod_bn",
        },
        "conformer_01_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_1_dropout_linear",
        },
        "conformer_01_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_01_ffmod_1_dropout", "input_dropout"],
        },
        "conformer_01_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_01_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_01_ffmod_1_ln": {"class": "layer_norm", "from": "input_dropout"},
        "conformer_01_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_2_dropout_linear",
        },
        "conformer_01_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_01_ffmod_2_dropout", "conformer_01_mhsa_mod_res_add"],
        },
        "conformer_01_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_01_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_01_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_01_mhsa_mod_res_add"},
        "conformer_01_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_01_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_01_mhsa_mod_att_linear"},
        "conformer_01_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_01_conv_mod_res_add"},
        "conformer_01_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_01_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_01_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_01_mhsa_mod_dropout", "conformer_01_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_01_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_01_mhsa_mod_ln",
            "key_shift": "conformer_01_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_01_output": {"class": "layer_norm", "from": "conformer_01_ffmod_2_half_res_add"},
        "conformer_02_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_02_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_02_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_02_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_02_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_conv_mod_pointwise_conv_2",
        },
        "conformer_02_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_02_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_02_conv_mod_ln": {"class": "layer_norm", "from": "conformer_02_ffmod_1_half_res_add"},
        "conformer_02_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_02_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_02_conv_mod_dropout", "conformer_02_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_02_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_02_conv_mod_bn",
        },
        "conformer_02_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_1_dropout_linear",
        },
        "conformer_02_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_02_ffmod_1_dropout", "conformer_01_output"],
        },
        "conformer_02_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_02_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_02_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_01_output"},
        "conformer_02_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_2_dropout_linear",
        },
        "conformer_02_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_02_ffmod_2_dropout", "conformer_02_mhsa_mod_res_add"],
        },
        "conformer_02_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_02_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_02_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_02_mhsa_mod_res_add"},
        "conformer_02_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_02_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_02_mhsa_mod_att_linear"},
        "conformer_02_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_02_conv_mod_res_add"},
        "conformer_02_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_02_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_02_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_02_mhsa_mod_dropout", "conformer_02_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_02_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_02_mhsa_mod_ln",
            "key_shift": "conformer_02_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_02_output": {"class": "layer_norm", "from": "conformer_02_ffmod_2_half_res_add"},
        "conformer_03_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_03_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_03_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_03_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_03_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_conv_mod_pointwise_conv_2",
        },
        "conformer_03_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_03_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_03_conv_mod_ln": {"class": "layer_norm", "from": "conformer_03_ffmod_1_half_res_add"},
        "conformer_03_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_03_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_03_conv_mod_dropout", "conformer_03_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_03_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_03_conv_mod_bn",
        },
        "conformer_03_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_1_dropout_linear",
        },
        "conformer_03_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_03_ffmod_1_dropout", "conformer_02_output"],
        },
        "conformer_03_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_03_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_03_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_02_output"},
        "conformer_03_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_2_dropout_linear",
        },
        "conformer_03_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_03_ffmod_2_dropout", "conformer_03_mhsa_mod_res_add"],
        },
        "conformer_03_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_03_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_03_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_03_mhsa_mod_res_add"},
        "conformer_03_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_03_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_03_mhsa_mod_att_linear"},
        "conformer_03_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_03_conv_mod_res_add"},
        "conformer_03_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_03_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_03_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_03_mhsa_mod_dropout", "conformer_03_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_03_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_03_mhsa_mod_ln",
            "key_shift": "conformer_03_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_03_output": {"class": "layer_norm", "from": "conformer_03_ffmod_2_half_res_add"},
        "conformer_04_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_04_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_04_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_04_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_04_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_conv_mod_pointwise_conv_2",
        },
        "conformer_04_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_04_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_04_conv_mod_ln": {"class": "layer_norm", "from": "conformer_04_ffmod_1_half_res_add"},
        "conformer_04_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_04_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_04_conv_mod_dropout", "conformer_04_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_04_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_04_conv_mod_bn",
        },
        "conformer_04_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_1_dropout_linear",
        },
        "conformer_04_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_04_ffmod_1_dropout", "conformer_03_output"],
        },
        "conformer_04_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_04_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_04_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_03_output"},
        "conformer_04_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_2_dropout_linear",
        },
        "conformer_04_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_04_ffmod_2_dropout", "conformer_04_mhsa_mod_res_add"],
        },
        "conformer_04_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_04_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_04_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_04_mhsa_mod_res_add"},
        "conformer_04_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_04_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_04_mhsa_mod_att_linear"},
        "conformer_04_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_04_conv_mod_res_add"},
        "conformer_04_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_04_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_04_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_04_mhsa_mod_dropout", "conformer_04_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_04_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_04_mhsa_mod_ln",
            "key_shift": "conformer_04_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_04_output": {"class": "layer_norm", "from": "conformer_04_ffmod_2_half_res_add"},
        "conformer_05_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_05_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_05_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_05_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_05_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_conv_mod_pointwise_conv_2",
        },
        "conformer_05_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_05_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_05_conv_mod_ln": {"class": "layer_norm", "from": "conformer_05_ffmod_1_half_res_add"},
        "conformer_05_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_05_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_05_conv_mod_dropout", "conformer_05_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_05_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_05_conv_mod_bn",
        },
        "conformer_05_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_1_dropout_linear",
        },
        "conformer_05_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_05_ffmod_1_dropout", "conformer_04_output"],
        },
        "conformer_05_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_05_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_05_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_04_output"},
        "conformer_05_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_2_dropout_linear",
        },
        "conformer_05_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_05_ffmod_2_dropout", "conformer_05_mhsa_mod_res_add"],
        },
        "conformer_05_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_05_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_05_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_05_mhsa_mod_res_add"},
        "conformer_05_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_05_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_05_mhsa_mod_att_linear"},
        "conformer_05_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_05_conv_mod_res_add"},
        "conformer_05_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_05_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_05_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_05_mhsa_mod_dropout", "conformer_05_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_05_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_05_mhsa_mod_ln",
            "key_shift": "conformer_05_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_05_output": {"class": "layer_norm", "from": "conformer_05_ffmod_2_half_res_add"},
        "conformer_06_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_06_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_06_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_06_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_06_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_conv_mod_pointwise_conv_2",
        },
        "conformer_06_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_06_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_06_conv_mod_ln": {"class": "layer_norm", "from": "conformer_06_ffmod_1_half_res_add"},
        "conformer_06_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_06_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_06_conv_mod_dropout", "conformer_06_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_06_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_06_conv_mod_bn",
        },
        "conformer_06_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_1_dropout_linear",
        },
        "conformer_06_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_06_ffmod_1_dropout", "conformer_05_output"],
        },
        "conformer_06_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_06_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_06_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_05_output"},
        "conformer_06_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_2_dropout_linear",
        },
        "conformer_06_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_06_ffmod_2_dropout", "conformer_06_mhsa_mod_res_add"],
        },
        "conformer_06_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_06_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_06_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_06_mhsa_mod_res_add"},
        "conformer_06_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_06_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_06_mhsa_mod_att_linear"},
        "conformer_06_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_06_conv_mod_res_add"},
        "conformer_06_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_06_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_06_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_06_mhsa_mod_dropout", "conformer_06_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_06_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_06_mhsa_mod_ln",
            "key_shift": "conformer_06_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_06_output": {"class": "layer_norm", "from": "conformer_06_ffmod_2_half_res_add"},
        "conformer_07_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_07_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_07_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_07_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_07_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_conv_mod_pointwise_conv_2",
        },
        "conformer_07_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_07_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_07_conv_mod_ln": {"class": "layer_norm", "from": "conformer_07_ffmod_1_half_res_add"},
        "conformer_07_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_07_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_07_conv_mod_dropout", "conformer_07_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_07_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_07_conv_mod_bn",
        },
        "conformer_07_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_1_dropout_linear",
        },
        "conformer_07_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_07_ffmod_1_dropout", "conformer_06_output"],
        },
        "conformer_07_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_07_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_07_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_06_output"},
        "conformer_07_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_2_dropout_linear",
        },
        "conformer_07_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_07_ffmod_2_dropout", "conformer_07_mhsa_mod_res_add"],
        },
        "conformer_07_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_07_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_07_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_07_mhsa_mod_res_add"},
        "conformer_07_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_07_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_07_mhsa_mod_att_linear"},
        "conformer_07_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_07_conv_mod_res_add"},
        "conformer_07_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_07_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_07_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_07_mhsa_mod_dropout", "conformer_07_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_07_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_07_mhsa_mod_ln",
            "key_shift": "conformer_07_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_07_output": {"class": "layer_norm", "from": "conformer_07_ffmod_2_half_res_add"},
        "conformer_08_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_08_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_08_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_08_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_08_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_conv_mod_pointwise_conv_2",
        },
        "conformer_08_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_08_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_08_conv_mod_ln": {"class": "layer_norm", "from": "conformer_08_ffmod_1_half_res_add"},
        "conformer_08_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_08_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_08_conv_mod_dropout", "conformer_08_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_08_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_08_conv_mod_bn",
        },
        "conformer_08_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_1_dropout_linear",
        },
        "conformer_08_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_08_ffmod_1_dropout", "conformer_07_output"],
        },
        "conformer_08_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_08_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_08_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_07_output"},
        "conformer_08_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_2_dropout_linear",
        },
        "conformer_08_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_08_ffmod_2_dropout", "conformer_08_mhsa_mod_res_add"],
        },
        "conformer_08_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_08_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_08_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_08_mhsa_mod_res_add"},
        "conformer_08_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_08_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_08_mhsa_mod_att_linear"},
        "conformer_08_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_08_conv_mod_res_add"},
        "conformer_08_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_08_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_08_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_08_mhsa_mod_dropout", "conformer_08_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_08_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_08_mhsa_mod_ln",
            "key_shift": "conformer_08_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_08_output": {"class": "layer_norm", "from": "conformer_08_ffmod_2_half_res_add"},
        "conformer_09_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_09_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_09_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_09_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_09_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_conv_mod_pointwise_conv_2",
        },
        "conformer_09_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_09_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_09_conv_mod_ln": {"class": "layer_norm", "from": "conformer_09_ffmod_1_half_res_add"},
        "conformer_09_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_09_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_09_conv_mod_dropout", "conformer_09_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_09_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_09_conv_mod_bn",
        },
        "conformer_09_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_1_dropout_linear",
        },
        "conformer_09_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_09_ffmod_1_dropout", "conformer_08_output"],
        },
        "conformer_09_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_09_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_09_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_08_output"},
        "conformer_09_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_2_dropout_linear",
        },
        "conformer_09_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_09_ffmod_2_dropout", "conformer_09_mhsa_mod_res_add"],
        },
        "conformer_09_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_09_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_09_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_09_mhsa_mod_res_add"},
        "conformer_09_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_09_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_09_mhsa_mod_att_linear"},
        "conformer_09_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_09_conv_mod_res_add"},
        "conformer_09_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_09_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_09_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_09_mhsa_mod_dropout", "conformer_09_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_09_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_09_mhsa_mod_ln",
            "key_shift": "conformer_09_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_09_output": {"class": "layer_norm", "from": "conformer_09_ffmod_2_half_res_add"},
        "conformer_10_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_10_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_10_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_10_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_10_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_conv_mod_pointwise_conv_2",
        },
        "conformer_10_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_10_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_10_conv_mod_ln": {"class": "layer_norm", "from": "conformer_10_ffmod_1_half_res_add"},
        "conformer_10_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_10_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_10_conv_mod_dropout", "conformer_10_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_10_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_10_conv_mod_bn",
        },
        "conformer_10_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_1_dropout_linear",
        },
        "conformer_10_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_10_ffmod_1_dropout", "conformer_09_output"],
        },
        "conformer_10_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_10_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_10_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_09_output"},
        "conformer_10_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_2_dropout_linear",
        },
        "conformer_10_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_10_ffmod_2_dropout", "conformer_10_mhsa_mod_res_add"],
        },
        "conformer_10_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_10_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_10_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_10_mhsa_mod_res_add"},
        "conformer_10_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_10_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_10_mhsa_mod_att_linear"},
        "conformer_10_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_10_conv_mod_res_add"},
        "conformer_10_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_10_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_10_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_10_mhsa_mod_dropout", "conformer_10_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_10_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_10_mhsa_mod_ln",
            "key_shift": "conformer_10_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_10_output": {"class": "layer_norm", "from": "conformer_10_ffmod_2_half_res_add"},
        "conformer_11_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_11_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_11_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_11_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_11_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_conv_mod_pointwise_conv_2",
        },
        "conformer_11_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_11_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_11_conv_mod_ln": {"class": "layer_norm", "from": "conformer_11_ffmod_1_half_res_add"},
        "conformer_11_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_11_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_11_conv_mod_dropout", "conformer_11_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_11_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_11_conv_mod_bn",
        },
        "conformer_11_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_1_dropout_linear",
        },
        "conformer_11_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_11_ffmod_1_dropout", "conformer_10_output"],
        },
        "conformer_11_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_11_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_11_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_10_output"},
        "conformer_11_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_2_dropout_linear",
        },
        "conformer_11_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_11_ffmod_2_dropout", "conformer_11_mhsa_mod_res_add"],
        },
        "conformer_11_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_11_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_11_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_11_mhsa_mod_res_add"},
        "conformer_11_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_11_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_11_mhsa_mod_att_linear"},
        "conformer_11_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_11_conv_mod_res_add"},
        "conformer_11_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_11_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_11_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_11_mhsa_mod_dropout", "conformer_11_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_11_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_11_mhsa_mod_ln",
            "key_shift": "conformer_11_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_11_output": {"class": "layer_norm", "from": "conformer_11_ffmod_2_half_res_add"},
        "conformer_12_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_12_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_12_conv_mod_depthwise_conv": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_12_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_12_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_conv_mod_pointwise_conv_2",
        },
        "conformer_12_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_12_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_12_conv_mod_ln": {"class": "layer_norm", "from": "conformer_12_ffmod_1_half_res_add"},
        "conformer_12_conv_mod_pointwise_conv_1": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_12_conv_mod_pointwise_conv_2": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_12_conv_mod_dropout", "conformer_12_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_12_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_12_conv_mod_bn",
        },
        "conformer_12_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_1_dropout_linear",
        },
        "conformer_12_ffmod_1_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_12_ffmod_1_dropout", "conformer_11_output"],
        },
        "conformer_12_ffmod_1_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_12_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_12_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_11_output"},
        "conformer_12_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_2_dropout_linear",
        },
        "conformer_12_ffmod_2_dropout_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_12_ffmod_2_dropout", "conformer_12_mhsa_mod_res_add"],
        },
        "conformer_12_ffmod_2_linear_swish": {
            "L2": ZHOU_L2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_12_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_12_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_12_mhsa_mod_res_add"},
        "conformer_12_mhsa_mod_att_linear": {
            "L2": ZHOU_L2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_12_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_12_mhsa_mod_att_linear"},
        "conformer_12_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_12_conv_mod_res_add"},
        "conformer_12_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_12_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_12_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_12_mhsa_mod_dropout", "conformer_12_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_12_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_12_mhsa_mod_ln",
            "key_shift": "conformer_12_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_12_output": {"class": "layer_norm", "from": "conformer_12_ffmod_2_half_res_add"},
        "enc_006": {  # for aux loss
            "class": "copy",
            "from": "conformer_06_output",
            "n_out": conf_model_dim,
        },
        out_layer_name: {
            "class": "copy",
            "from": "conformer_12_output",
            "n_out": conf_model_dim,
        },
    }
    network = augment_net_with_label_pops(
        network,
        label_info=label_info,
        classes_subsampling_info=SubsamplingInfo(factor=ss_factor, time_tag_name=time_tag_name),
    )
    network = {
        **network,
        "classes_": {
            **network["classes_"],
            "from": "slice_classes1",
            "set_dim_tags": {
                "T": returnn.CodeWrapper(f"{time_tag_name}.ceildiv_right({ss_factor//2}).ceildiv_right({ss_factor//2})")
            },
        },
        "slice_classes0": {
            "axis": "T",
            "class": "slice",
            "from": "data:classes",
            "slice_step": ss_factor // 2,
        },
        "slice_classes1": {
            "axis": "T",
            "class": "slice",
            "from": "slice_classes0",
            "slice_step": ss_factor // 2,
        },
    }
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=CONF_FOCAL_LOSS,
        l2=ZHOU_L2,
        label_info=label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        time_tag_name=time_tag_name,
        upsampling=False,
    )
    config = returnn.ReturnnConfig(
        config={
            "batching": "random",
            "batch_size": 15_000,
            "cache_size": "0",
            "chunking": "256:128",
            "debug_print_layer_output_template": True,
            "extern_data": {
                "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
                **extern_data.get_extern_data_config(label_info=label_info, time_tag_name=None),
            },
            "gradient_clip": 20,
            "gradient_noise": 0.0,
            "learning_rate": 0.001,
            "min_learning_rate": 1e-6,
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_error_measure": "sum_dev_score",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "log_batch_size": True,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 20,
            "newbob_multi_update_interval": 1,
            "optimizer": {"class": "nadam"},
            "optimizer_epsilon": 1e-8,
            "max_seqs": 128,
            "network": network,
            "tf_log_memory_usage": True,
            "use_tensorflow": True,
            "update_on_device": True,
            "window": 1,
        },
        hash_full_python_code=True,
        python_epilog=[
            _mask,
            random_mask,
            summary,
            transform,
            dynamic_learning_rate,
        ],
    )
    return config


# for debug only
def summary(name, x):
    """
    :param str name:
    :param tf.Tensor x: (batch,time,feature)
    """
    import tensorflow as tf

    # tf.summary.image wants [batch_size, height,  width, channels],
    # we have (batch, time, feature).
    img = tf.expand_dims(x, axis=3)  # (batch,time,feature,1)
    img = tf.transpose(img, [0, 2, 1, 3])  # (batch,feature,time,1)
    tf.summary.image(name, img, max_outputs=10)
    tf.summary.scalar("%s_max_abs" % name, tf.reduce_max(tf.abs(x)))
    mean = tf.reduce_mean(x)
    tf.summary.scalar("%s_mean" % name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    tf.summary.scalar("%s_stddev" % name, stddev)
    tf.summary.histogram("%s_hist" % name, tf.reduce_max(tf.abs(x), axis=2))


def _mask(x, batch_axis, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    """
    import tensorflow as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random.uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    from TFUtil import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    """
    import tensorflow as tf

    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.math.log(-tf.math.log(tf.random.uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
    else:
        _, x = tf.compat.v1.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.compat.v1.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims),
                    x,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def transform(data, network):
    # to be adjusted (20-50%)
    max_time_num = 1
    max_time = 15

    max_feature_num = 5
    max_feature = 5

    # halved before this step
    conservatvie_step = 2000

    x = data.placeholder
    import tensorflow as tf

    # summary("features", x)
    step = network.global_train_step
    increase_flag = tf.compat.v1.where(tf.greater_equal(step, conservatvie_step), 0, 1)

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=0,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // int(1 / 0.70 * max_time), max_time_num)
            // (1 + increase_flag),
            max_dims=max_time,
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=0,
            max_num=max_feature_num // (1 + increase_flag),
            max_dims=max_feature,
        )
        # summary("features_mask", x_masked)
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


# one cycle LR: triangular linear w.r.t. iterations(steps)
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    # -- need to be adjusted w.r.t. training -- #
    initialLR = 8e-5
    peakLR = 8e-4
    finalLR = 1e-6
    cycleEpoch = 180
    totalEpoch = 400
    nStep = 2420  # steps/epoch depending on batch_size

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
