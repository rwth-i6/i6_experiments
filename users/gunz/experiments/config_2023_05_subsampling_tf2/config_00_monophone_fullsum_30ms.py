__all__ = ["run", "run_single"]

import copy
import dataclasses
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
from ...setups.common.nn import baum_welch, oclr, returnn_time_tag
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
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
RASR_BINARY_PATH_TF = tk.Path(os.path.join(RASR_ROOT_TF2, "arch", RASR_ARCH), hash_overwrite="RASR_BINARY_PATH_TF")
RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON, hash_overwrite="RETURNN_PYTHON_EXE")

train_key = "train-other-960"


@dataclass(frozen=True, eq=True)
class Experiment:
    alignment_name: str
    bw_label_scale: float
    feature_stacking: bool
    feature_time_shift: float
    lr: str
    multitask: bool
    dc_detection: bool
    subsampling_factor: int

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    configs = [
        *(
            Experiment(
                alignment_name="scratch",
                bw_label_scale=0.3,
                dc_detection=False,
                feature_stacking=fs,
                feature_time_shift=10 / 1000,
                lr="v6",
                multitask=False,
                subsampling_factor=3,
            )
            for fs in [False]
        ),
        *(
            Experiment(
                alignment_name="scratch",
                bw_label_scale=0.3,
                dc_detection=False,
                feature_stacking=fs,
                feature_time_shift=7.5 / 1000,
                lr="v6",
                multitask=False,
                subsampling_factor=4,
            )
            for fs in [False]
        ),
    ]
    experiments = {
        exp: run_single(
            alignment_name=exp.alignment_name,
            bw_label_scale=exp.bw_label_scale,
            dc_detection=exp.dc_detection,
            feature_stacking=exp.feature_stacking,
            feature_time_shift=exp.feature_time_shift,
            focal_loss=exp.focal_loss,
            lr=exp.lr,
            multitask=exp.multitask,
            returnn_root=returnn_root,
            subsampling_factor=exp.subsampling_factor,
        )
        for exp in configs
    }

    return experiments


def run_single(
    *,
    alignment_name: str,
    bw_label_scale: float,
    dc_detection: bool,
    feature_stacking: bool,
    feature_time_shift: float,
    focal_loss: float,
    lr: str,
    multitask: bool,
    returnn_root: tk.Path,
    subsampling_factor: int,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-1-lr:{lr}-ss:{'fs' if feature_stacking else 'mp'}:{subsampling_factor}-bw:{bw_label_scale}"
    print(f"fh {name}")

    # ***********Initial arguments and init step ********************
    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = lbs_data_setups.get_data_inputs()

    rasr_init_args = lbs_data_setups.get_init_args(gt_normalization=True, dc_detection=dc_detection)
    rasr_init_args.feature_extraction_args["gt"]["parallel"] = 50
    rasr_init_args.feature_extraction_args["gt"]["rtf"] = 0.5
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
        chunk_size=CONF_CHUNKING_10MS,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="heuristic",
        eval_tdp_type="heuristic",
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
            "feature_stacking": feature_stacking,
            "reduction_factor": (1, subsampling_factor) if not feature_stacking else None,
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
        use_multi_task=multitask,
    )
    network["center-output"]["n_out"] = s.label_info.get_n_state_classes()

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "batch_size": 6144,
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
        "extern_data": {
            "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
        },
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

    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", remove_label_pops_and_losses_from_returnn_config(returnn_config))

    train_cfg = baum_welch.augment_for_fast_bw(
        crp=s.crp[s.crp_names["train"]],
        log_linear_scales=baum_welch.BwScales(
            label_posterior_scale=bw_label_scale, label_prior_scale=None, transition_scale=bw_label_scale
        ),
        returnn_config=returnn_config,
    )

    train_args = {
        **s.initial_train_args,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "returnn_config": copy.deepcopy(train_cfg),
    }
    train_j = s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        on_2080=False,
        include_alignment=False,
    )
    s.set_mono_priors_returnn_rasr(
        key="fh",
        epoch=keep_epochs[-2],
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        smoothen=True,
        returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
    )

    s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.triphone)
    s._update_crp_am_setting(crp_key="dev-other", tdp_type="default", add_base_allophones=False)

    decoding_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
    decoding_config.config["network"]["center-output"]["register_as_extern_data"] = "center-output"
    s.set_graph_for_experiment("fh", decoding_config)

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_BINARY_PATH_TF)

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
            lm_gc_simple_hash=True,
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
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                rtf_cpu=4,
            )

    if subsampling_factor == 3:
        for tdp_scale in [1.0]:
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
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
                lm_gc_simple_hash=True,
            )
            sil_tdp = (*recog_args.tdp_silence[:3], 3.0)
            align_cfg = (
                recog_args.with_prior_scale(0.6)
                .with_tdp_scale(tdp_scale)
                .with_tdp_silence(sil_tdp)
                .with_tdp_non_word(sil_tdp)
            )
            align_search_jobs = recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=align_cfg,
                num_encoder_output=conf_model_dim,
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
            crp.segment_path = corpus.SegmentCorpusJob(
                s.corpora[s.train_key].corpus_file, crp.concurrent
            ).out_segment_path

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
                segments=[
                    "train-other-960/2920-156224-0013/2920-156224-0013",
                    "train-other-960/2498-134786-0003/2498-134786-0003",
                    "train-other-960/6178-86034-0008/6178-86034-0008",
                    "train-other-960/5983-39669-0034/5983-39669-0034",
                ],
                show_labels=False,
                monophone=True,
            )
            tk.register_output(f"alignments/{name}/alignment-plots", plots.out_plot_folder)

            phoneme_durs = PlotPhonemeDurationsJob(
                alignment_bundle_path=a_job.out_alignment_bundle,
                allophones_path=allophones.out_allophone_file,
                time_step_s=30 / 1000,
            )
            tk.register_output(f"alignments/{name}/statistics/plots", phoneme_durs.out_plot_folder)
            tk.register_output(f"alignments/{name}/statistics/means", phoneme_durs.out_means)
            tk.register_output(f"alignments/{name}/statistics/variances", phoneme_durs.out_vars)

        for crp_k in ["dev-other", "dev-clean", "test-other", "test-clean"]:
            s.set_binaries_for_crp(crp_k, RASR_BINARY_PATH_TF)

            recognizer, recog_args = s.get_recognizer_and_args(
                key="fh",
                context_type=PhoneticContext.monophone,
                crp_corpus=crp_k,
                epoch=600,
                gpu=False,
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
                lm_gc_simple_hash=True,
            )
            sil_tdp = (*recog_args.tdp_silence[:3], 3.0)
            align_cfg = (
                recog_args.with_prior_scale(0.6)
                .with_tdp_scale(1.0)
                .with_tdp_silence(sil_tdp)
                .with_tdp_non_word(sil_tdp)
            )
            align_search_jobs = recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=align_cfg,
                num_encoder_output=conf_model_dim,
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
            crp.concurrent = 20

            recognizer.align(
                f"{name}-{crp_k}-pC{align_cfg.prior_info.center_state_prior.scale}-tdp{align_cfg.tdp_scale}",
                crp=crp,
                feature_scorer=align_search_jobs.search_feature_scorer,
                default_tdp=True,
                set_do_not_normalize_lemma_sequence_scores=False,
                rtf=2,
            )

            allophones = lexicon.StoreAllophonesJob(crp)
            tk.register_output(f"allophones/{name}/allophones", allophones.out_allophone_file)

    return s
