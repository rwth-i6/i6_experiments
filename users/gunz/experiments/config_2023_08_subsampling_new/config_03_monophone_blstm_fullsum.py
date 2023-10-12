__all__ = ["run", "run_single"]

import copy
import dataclasses
import re
import typing
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import corpus, lexicon, rasr, returnn
import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.analysis import (
    ComputeTimestampErrorJob,
    ComputeWordLevelTimestampErrorJob,
    PlotPhonemeDurationsJob,
    PlotViterbiAlignmentsJob,
)
from ...setups.common.nn import baum_welch, oclr
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    ALIGN_GMM_TRI_10MS,
    ALIGN_GMM_TRI_ALLOPHONES,
    CONF_CHUNKING_10MS,
    CONF_SA_CONFIG,
    BLSTM_FH_DECODING_TENSOR_CONFIG,
    L2,
    RASR_ARCH,
    RASR_ROOT_NO_TF,
    RASR_ROOT_TF2,
    RETURNN_PYTHON,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_NO_TF, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH")
RASR_BINARY_PATH_TF = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True, eq=True)
class Experiment:
    adapt_transition_model_to_ss: bool
    alignment_name: str
    alignment_prior_scale_center: float
    bw_label_scale: float
    bw_transition_scale: float
    feature_time_shift: float
    lr: str
    model_dim: int
    n_states_per_phone: int
    output_time_step: float
    subsampling_approach: str


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    model_dim = 512

    configs = [
        # Feature Stacking
        Experiment(
            adapt_transition_model_to_ss=False,
            alignment_name="scratch",
            alignment_prior_scale_center=0.0,
            bw_label_scale=0.3,
            feature_time_shift=10 / 1000,
            lr="v8",
            model_dim=model_dim,
            n_states_per_phone=1,
            output_time_step=40 / 1000,
            subsampling_approach="fs:4",
            bw_transition_scale=0.3,
        ),
        # 30ms feature stacking model provided by Tina
        # Max Pooling
        # 30ms and 40ms max pooling models already trained in previous pipelines
        # Adapted TDPs
        Experiment(
            adapt_transition_model_to_ss=True,
            alignment_name="scratch",
            alignment_prior_scale_center=0.6,
            bw_label_scale=0.3,
            feature_time_shift=10 / 1000,
            lr="v8",
            model_dim=model_dim,
            n_states_per_phone=1,
            output_time_step=40 / 1000,
            subsampling_approach="mp:2@2+mp:2@4",
            bw_transition_scale=0.3,
        ),
        # P-HMM-S
        Experiment(
            adapt_transition_model_to_ss=True,  # doesn't matter here, but set for hash compat
            alignment_name="scratch",
            alignment_prior_scale_center=0.6,
            bw_label_scale=0.3,
            feature_time_shift=7.5 / 1000,
            lr="v8",
            model_dim=model_dim,
            n_states_per_phone=1,
            output_time_step=30 / 1000,
            subsampling_approach="mp:2@2+mp:2@4",
            bw_transition_scale=0.0,
        ),
        Experiment(
            adapt_transition_model_to_ss=True,  # doesn't matter here, but set for hash compat
            alignment_name="scratch",
            alignment_prior_scale_center=0.0,
            bw_label_scale=0.3,
            feature_time_shift=10 / 1000,
            lr="v8",
            model_dim=model_dim,
            n_states_per_phone=1,
            output_time_step=40 / 1000,
            subsampling_approach="mp:2@2+mp:2@4",
            bw_transition_scale=0.0,
        ),
        Experiment(
            adapt_transition_model_to_ss=True,  # doesn't matter here, but set for hash compat
            alignment_name="scratch",
            alignment_prior_scale_center=0.3,
            bw_label_scale=0.3,
            feature_time_shift=10 / 1000,
            lr="v8",
            model_dim=model_dim,
            n_states_per_phone=1,
            output_time_step=40 / 1000,
            subsampling_approach="mp:2@2+mp:2@4",
            bw_transition_scale=0.0,
        ),
        Experiment(
            adapt_transition_model_to_ss=True,  # doesn't matter here, but set for hash compat
            alignment_name="scratch",
            alignment_prior_scale_center=0.6,
            bw_label_scale=0.3,
            feature_time_shift=10 / 1000,
            lr="v8",
            model_dim=model_dim,
            n_states_per_phone=1,
            output_time_step=40 / 1000,
            subsampling_approach="mp:2@2+mp:2@4",
            bw_transition_scale=0.0,
        ),
    ]
    experiments = {
        exp: run_single(
            adapt_transition_model_to_ss=exp.adapt_transition_model_to_ss,
            alignment_name=exp.alignment_name,
            align_prior_scale_center=exp.alignment_prior_scale_center,
            bw_label_scale=exp.bw_label_scale,
            feature_time_shift=exp.feature_time_shift,
            lr=exp.lr,
            model_dim=exp.model_dim,
            n_states_per_phone=exp.n_states_per_phone,
            returnn_root=returnn_root,
            output_time_step=exp.output_time_step,
            subsampling_approach=exp.subsampling_approach,
            bw_transition_scale=exp.bw_transition_scale,
        )
        for exp in configs
    }

    return experiments


def run_single(
    *,
    adapt_transition_model_to_ss: bool,
    alignment_name: str,
    align_prior_scale_center: float,
    bw_label_scale: float,
    bw_transition_scale: float,
    feature_time_shift: float,
    lr: str,
    n_states_per_phone: int,
    returnn_root: tk.Path,
    subsampling_approach: str,
    output_time_step: float,
    model_dim: int,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"blstm-1-lr:{lr}-n:{n_states_per_phone}-ss:{subsampling_approach}-dx:{output_time_step/(10/1000)}-d:{model_dim}-bwl:{bw_label_scale}-bwt:{bw_transition_scale}"
    if adapt_transition_model_to_ss:
        name += "-tdp:adapted"
    else:
        name += "-tdp:classic"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()

    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=False)
    rasr_init_args.feature_extraction_args["gt"]["parallel"] = 50
    rasr_init_args.feature_extraction_args["gt"]["rtf"] = 3
    rasr_init_args.feature_extraction_args["gt"]["gt_options"]["tempint_shift"] = feature_time_shift

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

    s.label_info = dataclasses.replace(
        s.label_info, n_states_per_phone=n_states_per_phone, state_tying=RasrStateTying.monophone
    )
    s.lexicon_args["norm_pronunciation"] = False
    s.lm_gc_simple_hash = True
    s.train_key = train_key

    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
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
        train_tdp_type=f"heuristic-{int(output_time_step * 1000)}ms" if adapt_transition_model_to_ss else "heuristic",
        eval_tdp_type=f"heuristic-{int(output_time_step * 1000)}ms" if adapt_transition_model_to_ss else "heuristic",
        add_base_allophones=False,
    )

    # ---------------------- returnn config---------------
    partition_epochs = {"train": 40, "dev": 1}

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
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_bwd_2": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_1", "lstm_bwd_1"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_bwd_3": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_2", "lstm_bwd_2"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_bwd_4": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_3", "lstm_bwd_3"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_bwd_5": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_4", "lstm_bwd_4"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_bwd_6": {
            "L2": 0.01,
            "class": "rec",
            "direction": -1,
            "dropout": 0.1,
            "from": ["lstm_fwd_5", "lstm_bwd_5"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_fwd_1": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["source"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_fwd_2": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_1", "lstm_bwd_1"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_fwd_3": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_2", "lstm_bwd_2"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_fwd_4": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_3", "lstm_bwd_3"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_fwd_5": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_4", "lstm_bwd_4"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "lstm_fwd_6": {
            "L2": 0.01,
            "class": "rec",
            "direction": 1,
            "dropout": 0.1,
            "from": ["lstm_fwd_5", "lstm_bwd_5"],
            "n_out": model_dim,
            "unit": "nativelstm2",
        },
        "encoder-output": {
            "class": "copy",
            "from": ["lstm_fwd_6", "lstm_bwd_6"],
            "n_out": 2 * model_dim,
        },
    }
    network = augment_net_with_monophone_outputs(
        network,
        add_mlps=True,
        encoder_output_len=2 * model_dim,
        final_ctx_type=PhoneticContext.triphone_forward,
        focal_loss_factor=2.0,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=0.0,
        use_multi_task=False,
    )
    network["center-output"]["n_out"] = s.label_info.get_n_state_classes()

    assert subsampling_approach.count("fs:") <= 1, "can only feature stack once"
    for part in subsampling_approach.split("+"):
        if part.startswith("fs"):
            _, factor = part.split(":")
            factor = int(factor)

            network = {
                **network,
                "feature_stacking": {
                    "class": "window",
                    "from": "source",
                    "stride": factor,
                    "window_left": factor - 1,
                    "window_right": 0,
                    "window_size": factor,
                },
                "feature_stacking_merged": {
                    "axes": "except_time",
                    "class": "merge_dims",
                    "from": "feature_stacking",
                },
                "lstm_fwd_1": {
                    **network["lstm_fwd_1"],
                    "from": "feature_stacking_merged",
                },
                "lstm_bwd_1": {
                    **network["lstm_bwd_1"],
                    "from": "feature_stacking_merged",
                },
            }
        elif part.startswith("mp"):
            match = re.match(r"^mp:(\d+)@(\d+)$", part)

            assert match is not None, f"syntax error: {part}"
            factor, layer = match.groups()

            l_name = f"mp-{layer}"
            network[l_name] = {
                "class": "pool",
                "from": network[f"lstm_fwd_{layer}"]["from"],
                "mode": "max",
                "padding": "same",
                "pool_size": (int(factor),),
            }
            network[f"lstm_fwd_{layer}"]["from"] = l_name
            network[f"lstm_bwd_{layer}"]["from"] = l_name
        else:
            assert False, f"unknown subsampling instruction {part}"

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
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {"data": {"dim": 50}},
    }
    keep_epochs = [400, 550, num_epochs]
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

    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)

    train_cfg = baum_welch.augment_for_fast_bw(
        crp=s.crp[s.crp_names["train"]],
        log_linear_scales=baum_welch.BwScales(
            label_posterior_scale=bw_label_scale, label_prior_scale=None, transition_scale=bw_transition_scale
        ),
        returnn_config=returnn_config,
    )

    train_args = {
        **s.initial_train_args,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "returnn_config": copy.deepcopy(train_cfg),
    }
    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        include_alignment=False,
    )

    s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.triphone)
    s._update_crp_am_setting(crp_key="dev-other", tdp_type="default", add_base_allophones=False)

    graph_cfg = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    graph_cfg.config["network"]["encoder-output"]["register_as_extern_data"] = "encoder-output"
    graph_cfg.config["network"]["center-output"]["register_as_extern_data"] = "center-output"

    s.set_returnn_config_for_experiment("fh", graph_cfg)
    s.set_graph_for_experiment("fh", override_cfg=graph_cfg)

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_BINARY_PATH_TF)

        prior_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)

        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=ep,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=prior_config,
        )

        base_params = s.get_cart_params("fh")
        monophone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.monophone)
        nn_pch_config = copy.deepcopy(prior_config)
        nn_pch_config.config["network"]["center-output"] = {
            **nn_pch_config.config["network"]["center-output"],
            "class": "linear",
            "activation": "log_softmax",
            "register_as_extern_data": "center-output",
        }
        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "monophone-dense"

        for pC, tdp in itertools.product([0.4, 0.6], [0.4, 0.6]):
            cfg = dataclasses.replace(base_params, beam_limit=100000, lm_scale=1.5, tdp_scale=tdp).with_prior_scale(pC)
            s.recognize_cart(
                key="fh",
                crp_corpus="dev-other",
                epoch=max(keep_epochs),
                calculate_statistics=True,
                cart_tree_or_tying_config=tying_cfg,
                log_softmax_returnn_config=nn_pch_config,
                encoder_output_layer="center__output",
                n_cart_out=monophone_li.get_n_of_dense_classes(),
                params=cfg,
                lm_gc_simple_hash=True,
                opt_lm_am_scale=True,
                cpu_rqmt=2,
                mem_rqmt=4,
                rtf=12,
                native_ops=["NativeLstm2"],
            )

    s.set_binaries_for_crp("train-other-960.train", RASR_BINARY_PATH_TF)
    s.create_stm_from_corpus("train-other-960.train")
    s._set_scorer_for_corpus("train-other-960.train")
    s._init_lm("train-other-960.train", **next(iter(dev_data_inputs.values())).lm)
    s._update_crp_am_setting("train-other-960.train", tdp_type="default", add_base_allophones=False)
    recognizer, recog_args = s.get_recognizer_and_args(
        key="fh",
        context_type=PhoneticContext.monophone,
        crp_corpus="train-other-960.train",
        epoch=600,
        gpu=False,
        tensor_map=BLSTM_FH_DECODING_TENSOR_CONFIG,
        set_batch_major_for_feature_scorer=False,
        tf_library=s.native_lstm2_job.out_op,
        lm_gc_simple_hash=True,
    )

    tdp_scale = 0.0 if bw_transition_scale == 0.0 else 1.0
    sil_tdp = (0.0, 3.0, "infinity", 0.0)
    align_cfg = (
        recog_args.with_prior_scale(align_prior_scale_center)
        .with_tdp_scale(tdp_scale)
        .with_tdp_speech((3.0, 0.0, "infinity", 0.0))
        .with_tdp_silence(sil_tdp)
        .with_tdp_non_word(sil_tdp)
    )
    align_search_jobs = recognizer.recognize_count_lm(
        label_info=s.label_info,
        search_parameters=align_cfg,
        num_encoder_output=2 * model_dim,
        rerun_after_opt_lm=False,
        opt_lm_am=False,
        add_sis_alias_and_output=False,
        calculate_stats=True,
        rtf_cpu=4,
    )
    crp = copy.deepcopy(align_search_jobs.search_crp)
    crp.acoustic_model_config.tdp.applicator_type = "corrected"
    crp.acoustic_model_config.allophones.add_all = False
    crp.acoustic_model_config.allophones.add_from_lexicon = True
    crp.concurrent = 300
    crp.segment_path = corpus.SegmentCorpusJob(s.corpora[s.train_key].corpus_file, crp.concurrent).out_segment_path

    a_name = f"{name}-pC{align_cfg.prior_info.center_state_prior.scale}-silE{align_cfg.tdp_silence[-1]}-tdp{align_cfg.tdp_scale}"
    a_job = recognizer.align(
        a_name,
        crp=crp,
        feature_scorer=align_search_jobs.search_feature_scorer,
        default_tdp=False,
        set_do_not_normalize_lemma_sequence_scores=False,
        rtf=4,
    )
    # a_job.alignment_flow.flags["cache_mode"] = "bundle"

    allophones = lexicon.StoreAllophonesJob(crp)
    tk.register_output(f"allophones/{a_name}/allophones", allophones.out_allophone_file)

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        segments=["train-other-960/2920-156224-0013/2920-156224-0013"],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/{a_name}/alignment-plots", plots.out_plot_folder)

    phoneme_durs = PlotPhonemeDurationsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        time_step_s=feature_time_shift * 4,
    )
    tk.register_output(f"alignments/{a_name}/statistics/plots", phoneme_durs.out_plot_folder)
    tk.register_output(f"alignments/{a_name}/statistics/means", phoneme_durs.out_means)
    tk.register_output(f"alignments/{a_name}/statistics/variances", phoneme_durs.out_vars)

    tse_job = ComputeTimestampErrorJob(
        allophones=allophones.out_allophone_file,
        alignment=a_job.out_alignment_bundle,
        t_step=feature_time_shift * 4,
        reference_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        reference_t_step=10 / 1000,
    )
    tse_job.add_alias(f"tse/{a_name}")
    tk.register_output(f"alignments/{a_name}/statistics/tse", tse_job.out_tse)

    tse_w_job = ComputeWordLevelTimestampErrorJob(
        allophones=allophones.out_allophone_file,
        alignment=a_job.out_alignment_bundle,
        t_step=output_time_step,
        reference_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        reference_t_step=10 / 1000,
    )
    tse_w_job.add_alias(f"tse-w/{a_name}/tse")
    tk.register_output(f"alignments/{a_name}/statistics/tse-w", tse_w_job.out_tse)

    s.experiments["fh"]["alignment_job"] = a_job

    return s
