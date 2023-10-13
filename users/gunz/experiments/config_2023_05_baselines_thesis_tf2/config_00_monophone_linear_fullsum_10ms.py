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

from ...setups.common.analysis import PlotPhonemeDurationsJob, PlotViterbiAlignmentsJob
from ...setups.common.nn import baum_welch, oclr
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network import augment
from ...setups.fh.network.augment import (
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING,
    CONF_SA_CONFIG,
    MLP_FH_DECODING_TENSOR_CONFIG,
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
    alignment_name: str
    bw_label_scale: float
    context_window_size: int
    lr: str
    model_dim: int
    output_time_step: float
    w_init: typing.Optional[str]


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    model_dim = 2048

    configs = [
        Experiment(
            alignment_name="scratch",
            bw_label_scale=0.3,
            context_window_size=15,
            lr="v8",
            model_dim=model_dim,
            output_time_step=10 / 1000,
            w_init=augment.DEFAULT_INIT,
        ),
        Experiment(
            alignment_name="scratch",
            bw_label_scale=0.3,
            context_window_size=15,
            lr="v8",
            model_dim=model_dim,
            output_time_step=10 / 1000,
            w_init=None,
        ),
    ]
    experiments = {
        exp: run_single(
            alignment_name=exp.alignment_name,
            bw_label_scale=exp.bw_label_scale,
            context_window_size=exp.context_window_size,
            lr=exp.lr,
            model_dim=exp.model_dim,
            output_time_step=exp.output_time_step,
            returnn_root=returnn_root,
            w_init=exp.w_init,
        )
        for exp in configs
    }

    return experiments


def run_single(
    *,
    alignment_name: str,
    bw_label_scale: float,
    context_window_size: int,
    lr: str,
    returnn_root: tk.Path,
    output_time_step: float,
    model_dim: int,
    w_init: typing.Optional[str],
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"mlp-1-lr:{lr}-dx:{output_time_step/(10/1000)}-d:{model_dim}-bw:{bw_label_scale}-w:{'default' if w_init == augment.DEFAULT_INIT else w_init}"
    print(f"fh {name}")

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

    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1, state_tying=RasrStateTying.monophone)
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
        chunk_size=CONF_CHUNKING,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="heuristic",
        eval_tdp_type="heuristic",
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
        "window": {
            "class": "window",
            "from": "source",
            "window_size": context_window_size,
        },
        "window-merged": {
            "axes": "except_time",
            "class": "merge_dims",
            "from": "window",
        },
        "linear-1": {
            "class": "linear",
            "activation": "relu",
            "from": "window-merged",
            "dropout": 0.1,
            "forward_weights_init": w_init,
            "L2": 0.0001,
            "n_out": model_dim,
        },
        "linear-2": {
            "class": "linear",
            "activation": "relu",
            "from": "linear-1",
            "dropout": 0.1,
            "forward_weights_init": w_init,
            "L2": 0.0001,
            "n_out": model_dim,
        },
        "linear-3": {
            "class": "linear",
            "activation": "relu",
            "from": "linear-2",
            "dropout": 0.1,
            "forward_weights_init": w_init,
            "L2": 0.0001,
            "n_out": model_dim,
        },
        "linear-4": {
            "class": "linear",
            "activation": "relu",
            "from": "linear-3",
            "dropout": 0.1,
            "forward_weights_init": w_init,
            "L2": 0.0001,
            "n_out": model_dim,
        },
        "linear-5": {
            "class": "linear",
            "activation": "relu",
            "from": "linear-4",
            "dropout": 0.1,
            "forward_weights_init": w_init,
            "L2": 0.0001,
            "n_out": model_dim,
        },
        "linear-6": {
            "class": "linear",
            "activation": "relu",
            "from": "linear-5",
            "dropout": 0.1,
            "forward_weights_init": w_init,
            "L2": 0.0001,
            "n_out": model_dim,
        },
        "encoder-output": {
            "class": "copy",
            "from": "linear-6",
            "register_as_extern_data": "encoder-output",
        },
        "center-output": {
            "class": "softmax",
            "n_out": s.label_info.get_n_of_dense_classes(),
            "from": "encoder-output",
            "forward_weights_init": augment.DEFAULT_INIT,
            "register_as_extern_data": "center-output",
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
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_clip": 10,
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
    s.set_returnn_config_for_experiment("fh", remove_label_pops_and_losses_from_returnn_config(returnn_config))

    train_cfg = baum_welch.augment_for_fast_bw(
        crp=s.crp[s.crp_names["train"]],
        log_linear_scales=baum_welch.BwScales(
            label_posterior_scale=bw_label_scale, label_prior_scale=None, transition_scale=0.1
        ),
        returnn_config=returnn_config,
    )

    train_args = {
        **s.initial_train_args,
        "time_rqmt": 60,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "returnn_config": copy.deepcopy(train_cfg),
    }
    s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        on_2080=False,
        include_alignment=False,
    )

    s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.triphone)
    s._update_crp_am_setting(crp_key="dev-other", tdp_type="default", add_base_allophones=False)
    s.set_graph_for_experiment("fh", override_cfg=remove_label_pops_and_losses_from_returnn_config(returnn_config))

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_BINARY_PATH_TF)

        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=ep,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
        )
        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=MLP_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
        )

        recog_args = recog_args.with_lm_scale(1.0).with_prior_scale(0.5)

        for pC, tdp_simple, tdp_scale in itertools.product([0.5], [False], [0.1, 0.2]):
            cfg = recog_args.with_prior_scale(pC).with_tdp_scale(tdp_scale)

            if tdp_simple:
                sil_non_w_tdp = (0.0, 0.0, "infinity", 20.0)
                cfg = dataclasses.replace(
                    cfg, tdp_non_word=sil_non_w_tdp, tdp_silence=sil_non_w_tdp, tdp_speech=(0.0, 0.0, "infinity", 0.0)
                )

            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                rtf_cpu=12,
            )

    tdp_scale = 1.0
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
        tensor_map=MLP_FH_DECODING_TENSOR_CONFIG,
        set_batch_major_for_feature_scorer=False,
        lm_gc_simple_hash=True,
    )
    sil_tdp = (*recog_args.tdp_silence[:3], 3.0)
    align_cfg = (
        recog_args.with_prior_scale(0.6).with_tdp_scale(tdp_scale).with_tdp_silence(sil_tdp).with_tdp_non_word(sil_tdp)
    )
    align_search_jobs = recognizer.recognize_count_lm(
        label_info=s.label_info,
        search_parameters=align_cfg,
        num_encoder_output=model_dim,
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

    a_job = recognizer.align(
        f"{name}-pC{align_cfg.prior_info.center_state_prior.scale}-tdp{align_cfg.tdp_scale}",
        crp=crp,
        feature_scorer=align_search_jobs.search_feature_scorer,
        default_tdp=True,
        set_do_not_normalize_lemma_sequence_scores=False,
    )

    allophones = lexicon.StoreAllophonesJob(crp)
    tk.register_output(f"allophones/{name}/allophones", allophones.out_allophone_file)

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        segments=["train-other-960/2920-156224-0013/2920-156224-0013"],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/{name}/alignment-plots", plots.out_plot_folder)

    phoneme_durs = PlotPhonemeDurationsJob(
        alignment_bundle_path=a_job.out_alignment_bundle,
        allophones_path=allophones.out_allophone_file,
        time_step_s=10 / 1000,
    )
    tk.register_output(f"alignments/{name}/statistics/plots", phoneme_durs.out_plot_folder)
    tk.register_output(f"alignments/{name}/statistics/means", phoneme_durs.out_means)
    tk.register_output(f"alignments/{name}/statistics/variances", phoneme_durs.out_vars)

    s.experiments["fh"]["alignment_job"] = a_job

    return s
