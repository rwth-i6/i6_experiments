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


from i6_core import corpus, lexicon, mm, rasr, returnn
from i6_core.features import FeatureExtractionJob, basic_cache_flow
from i6_core.rasr import PrecomputedHybridFeatureScorer
from i6_core.returnn.flow import add_tf_flow_to_base_flow, make_precomputed_hybrid_tf_feature_flow

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.analysis import (
    ComputeSilencePercentageJob,
    ComputeTimestampErrorJob,
    ComputeWordLevelTimestampErrorJob,
    PlotViterbiAlignmentsJob,
)
from ...setups.common.analysis.tse_tina import ComputeTinaTseJob
from ...setups.common.nn import baum_welch, oclr, returnn_time_tag, seq_disc
from ...setups.common.nn.chunking import subsample_chunking
from ...setups.common.nn.compile_graph import compile_tf_graph_from_returnn_config
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.network import conformer, diphone_joint_output
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    SubsamplingInfo,
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    ALIGN_GMM_TRI_10MS,
    ALIGN_GMM_TRI_ALLOPHONES,
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
    label_smoothing: float
    lr: str
    dc_detection: bool
    run_performance_study: bool
    tune_decoding: bool
    tune_nn_pch: bool
    run_tdp_study: bool

    filter_segments: typing.Optional[typing.List[str]] = None
    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path, alignment: tk.Path, a_name: str):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    exp = Experiment(
        alignment=alignment,
        alignment_name=a_name,
        batch_size=12500,
        chunking=CONF_CHUNKING_10MS,
        dc_detection=False,
        decode_all_corpora=False,
        fine_tune=a_name in ["40ms-FF-v8", "40ms-FFs-v8"],
        label_smoothing=CONF_LABEL_SMOOTHING,
        lr="v13",
        run_performance_study=a_name == "40ms-FF-v8",
        tune_decoding=a_name in ["40ms-FFs-v8", "40ms-Bmp-pC0.6", "40ms-Bs-pC0.6"],  # ["40ms-FF-v8", "40ms-FFs-v8"],
        tune_nn_pch=a_name in ["40ms-FFs-v8"],
        run_tdp_study=False,
    )
    sys = run_single(
        alignment=exp.alignment,
        alignment_name=exp.alignment_name,
        batch_size=exp.batch_size,
        chunking=exp.chunking,
        dc_detection=exp.dc_detection,
        decode_all_corpora=exp.decode_all_corpora,
        fine_tune=exp.fine_tune,
        focal_loss=exp.focal_loss,
        label_smoothing=exp.label_smoothing,
        returnn_root=returnn_root,
        run_performance_study=exp.run_performance_study,
        tune_decoding=exp.tune_decoding,
        filter_segments=exp.filter_segments,
        lr=exp.lr,
        run_tdp_study=exp.run_tdp_study,
        tune_nn_pch=exp.tune_nn_pch,
    )

    return exp, sys


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
    lr: str,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    tune_nn_pch: bool,
    run_tdp_study: bool,
    label_smoothing: float = CONF_LABEL_SMOOTHING,
    filter_segments: typing.Optional[typing.List[str]],
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-2-a:{alignment_name}-lr:{lr}-fl:{focal_loss}-ls:{label_smoothing}-ch:{chunking}"
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

    time_prolog, time_tag_name = returnn_time_tag.get_shared_time_tag()
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=chunking,
        focal_loss_factor=CONF_FOCAL_LOSS,
        label_smoothing=label_smoothing,
        num_classes=s.label_info.get_n_of_dense_classes(),
        time_tag_name=time_tag_name,
        upsample_by_transposed_conv=False,
        conf_args={
            "feature_stacking": False,
            "reduction_factor": (1, ss_factor),
        },
    )
    network = network_builder.network
    network = augment_net_with_label_pops(
        network,
        label_info=s.label_info,
        classes_subsampling_info=SubsamplingInfo(factor=ss_factor, time_tag_name=time_tag_name),
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
    if label_smoothing > 0:
        # Make sure it is defined
        network["linear1-triphone"]["from"] = ["encoder-output"]
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=0.0,  # no LS here!
        time_tag_name=time_tag_name,
        upsampling=False,
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

    for ep, crp_k in itertools.product([300, 400, 500, 550, 600], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        s.set_diphone_priors_returnn_rasr(
            key="fh",
            epoch=ep,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
        )

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
        recog_args = recog_args.with_lm_scale(round(recog_args.lm_scale / float(ss_factor), 2)).with_tdp_scale(0.1)

        # Top 3 from monophone TDP study
        good_values = [
            ((3, 0, "infinity", 0), (3, 10, "infinity", 10)),  # 8,9%
            ((3, 0, "infinity", 3), (3, 10, "infinity", 10)),  # 8,9%
            ((3, 0, "infinity", 0), (10, 10, "infinity", 10)),  # 9,0%
            *([((3, 0, "infinity", 0), (0, 3, "infinity", 20))] if ep == max(keep_epochs) else []),  # default
        ]

        for cfg in [
            recog_args.with_prior_scale(0.4, 0.4),
            recog_args.with_prior_scale(0.4, 0.2),
            recog_args.with_prior_scale(0.4, 0.6)
            .with_tdp_scale(0.4)
            .with_tdp_speech((3, 0, "infinity", 0))
            .with_tdp_silence((3, 10, "infinity", 10)),
            *(
                recog_args.with_prior_scale(0.4, 0.4)
                .with_tdp_scale(0.4)
                .with_tdp_speech(tdp_sp)
                .with_tdp_silence(tdp_sil)
                for tdp_sp, tdp_sil in good_values
            ),
        ]:
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                rtf_cpu=20,
            )

        if tune_decoding and ep == keep_epochs[-1]:
            best_config = recognizer.recognize_optimize_scales(
                label_info=s.label_info,
                search_parameters=recog_args.with_lm_scale(1.5),
                num_encoder_output=conf_model_dim,
                tdp_speech=[(3, 0, "infinity", 0)]
                if "Bmp" in alignment_name
                else [(3, 0, "infinity", 0), (0, 0, "infinity", 0)],
                tdp_sil=[(3, 10, "infinity", 10)]
                if "Bmp" in alignment_name
                else [(3, 10, "infinity", 10), (0, 3, "infinity", 20)],
                prior_scales=list(
                    itertools.product(
                        [round(v, 1) for v in np.linspace(0.2, 0.8, 4)],
                        [round(v, 1) for v in np.linspace(0.2, 0.6, 3)],
                    )
                ),
                tdp_scales=[0.4, 0.6] if "Bmp" in alignment_name else [round(v, 1) for v in np.linspace(0.2, 0.6, 3)],
            )
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=best_config,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                name_override="best/4gram",
                rtf_cpu=32,
            )

    if tune_nn_pch:
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
        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=keep_epochs[-2],
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=prior_returnn_config,
            output_layer_name="output",
        )

        diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "diphone-dense"
        base_params = dataclasses.replace(s.get_cart_params("fh"), beam_limit=100000, lm_scale=1.5)

        s.recognize_optimize_scales_nn_pch(
            key="fh",
            epoch=max(keep_epochs),
            cart_tree_or_tying_config=tying_cfg,
            crp_corpus="dev-other",
            log_softmax_returnn_config=nn_precomputed_returnn_config,
            n_out=diphone_li.get_n_of_dense_classes(),
            params=base_params,
            prior_scales=[round(v, 1) for v in np.linspace(0.2, 0.8, 4)],
            tdp_scales=[round(v, 1) for v in np.linspace(0.2, 0.8, 4)],
            tdp_speech=[(0, 0, "infinity", 0), (3, 0, "infinity", 0)],
        )

        for cfg in [base_params.with_lm_scale(1.5).with_tdp_scale(sc).with_prior_scale(sc) for sc in [0.4, 0.6]]:
            s.recognize_cart(
                key="fh",
                epoch=max(keep_epochs),
                calculate_statistics=True,
                cart_tree_or_tying_config=tying_cfg,
                cpu_rqmt=2,
                crp_corpus="dev-other",
                lm_gc_simple_hash=True,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                mem_rqmt=4,
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                opt_lm_am_scale=True,
                params=cfg,
            )

    if run_performance_study:
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
        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=keep_epochs[-2],
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
                s.get_cart_params("fh"), altas=a, beam=beam, beam_limit=100000, lm_scale=2, tdp_scale=0.4
            ).with_prior_scale(pC)
            for beam, pC, a in itertools.product(
                [14, 16, 18, 20],
                [0.4, 0.6],
                [None, 2, 4],
            )
        ]
        for cfg in configs:
            j = s.recognize_cart(
                key="fh",
                epoch=max(keep_epochs),
                calculate_statistics=True,
                cart_tree_or_tying_config=tying_cfg,
                cpu_rqmt=2,
                crp_corpus="dev-other",
                lm_gc_simple_hash=True,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                mem_rqmt=4,
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                params=cfg,
                opt_lm_am_scale=True,
                rtf=1.5,
            )
            j.rqmt.update({"sbatch_args": ["-p", "rescale_amd"]})

        configs = [
            dataclasses.replace(
                s.get_cart_params("fh"), beam=16, beam_limit=100000, lm_scale=2, tdp_scale=tdpS
            ).with_prior_scale(pC)
            for pC, tdpS in itertools.product(
                [round(v, 1) for v in np.linspace(0.2, 0.8, 4)],
                [round(v, 1) for v in np.linspace(0.2, 0.8, 4)],
            )
        ]

        sys2 = copy.deepcopy(s)
        sys2.lexicon_args["norm_pronunciation"] = False
        sys2._update_am_setting_for_all_crps(
            train_tdp_type="default",
            eval_tdp_type="default",
        )

        for cfg in configs:
            for sys in [s, sys2]:
                sys.recognize_cart(
                    key="fh",
                    epoch=max(keep_epochs),
                    calculate_statistics=True,
                    cart_tree_or_tying_config=tying_cfg,
                    cpu_rqmt=2,
                    crp_corpus="dev-other",
                    lm_gc_simple_hash=True,
                    alias_output_prefix="no-norm-pron/" if not sys.lexicon_args["norm_pronunciation"] else "",
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

        if alignment_name == "40ms-FFs-v8":
            configs = [
                (
                    8e-5,
                    True,
                    baum_welch.BwScales(label_posterior_scale=1.0, label_prior_scale=None, transition_scale=0.0),
                )
            ]
        else:
            configs = [
                (
                    8e-5,
                    False,
                    baum_welch.BwScales(label_posterior_scale=1.0, label_prior_scale=None, transition_scale=0.3),
                )
            ]

        peak_lr, adapt_transition_model, bw_scale = configs[0]

        ft_name = f"{orig_name}-fs:{peak_lr}-bwl:{bw_scale.label_posterior_scale}-bwt:{bw_scale.transition_scale}"
        s.set_experiment_dict("fh-fs", alignment_name, "di", postfix_name=ft_name)

        s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
        s.lexicon_args["norm_pronunciation"] = False

        s._update_am_setting_for_all_crps(
            train_tdp_type="heuristic-40ms" if adapt_transition_model else "heuristic",
            eval_tdp_type="heuristic-40ms" if adapt_transition_model else "heuristic",
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
        bw_train_job = s.returnn_rasr_training(
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
                    j.rqmt.update({"sbatch_args": ["-p", "rescale_amd"]})

        s.set_graph_for_experiment("fh-fs", override_cfg=nn_precomputed_returnn_config)

        s.set_binaries_for_crp("train-other-960.train", RASR_TF_BINARY_PATH)
        s.create_stm_from_corpus("train-other-960.train")
        s._set_scorer_for_corpus("train-other-960.train")
        s._init_lm("train-other-960.train", **next(iter(dev_data_inputs.values())).lm)
        s.label_info = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
        s._update_crp_am_setting("train-other-960.train", tdp_type="default", add_base_allophones=False)

        for sil_e in [3.0, 0.0]:  # reverse order to use 0.0 in other exps
            crp = copy.deepcopy(s.crp["train-other-960.train"])
            crp.acoustic_model_config.allophones.add_all = False
            crp.acoustic_model_config.allophones.add_from_lexicon = True
            crp.acoustic_model_config.state_tying.type = "diphone-dense"
            crp.acoustic_model_config.tdp.applicator_type = "corrected"
            crp.acoustic_model_config.tdp.scale = 1.0

            v = (3.0, 0.0, "infinity", 0.0)
            sv = (0.0, 3.0, "infinity", sil_e)
            keys = ["loop", "forward", "skip", "exit"]
            for i, k in enumerate(keys):
                crp.acoustic_model_config.tdp["*"][k] = v[i]
                crp.acoustic_model_config.tdp["silence"][k] = sv[i]

            crp.concurrent = 300
            crp.segment_path = corpus.SegmentCorpusJob(
                s.corpora[s.train_key].corpus_file, crp.concurrent
            ).out_segment_path

            p_mixtures = mm.CreateDummyMixturesJob(
                s.label_info.get_n_of_dense_classes(), s.initial_nn_args["num_input"]
            )
            priors = s.experiments["fh-fs"]["priors"].center_state_prior
            p_c = 0.6
            feature_scorer = PrecomputedHybridFeatureScorer(
                prior_mixtures=p_mixtures.out_mixtures,
                prior_file=priors.file,
                priori_scale=p_c,
            )
            tf_flow = make_precomputed_hybrid_tf_feature_flow(
                tf_graph=s.experiments["fh-fs"]["graph"]["inference"],
                tf_checkpoint=s.experiments["fh-fs"]["train_job"].out_checkpoints[fine_tune_epochs],
                output_layer_name="output",
                tf_fwd_input_name="tf-fwd-input",
            )
            feature_flow = add_tf_flow_to_base_flow(
                base_flow=s.feature_flows["train-other-960.train"],
                tf_flow=tf_flow,
                tf_fwd_input_name="tf-fwd-input",
            )

            a_name = f"{ft_name}-pC{p_c}-silE{sil_e}-tdp{1.0}"
            recognizer, _ = s.get_recognizer_and_args(
                key="fh-fs",
                context_type=PhoneticContext.diphone,
                crp_corpus="train-other-960.train",
                epoch=fine_tune_epochs,
                gpu=False,
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
                lm_gc_simple_hash=True,
            )
            a_job = recognizer.align(
                a_name,
                crp=crp,
                feature_scorer=feature_scorer,
                feature_flow=feature_flow,
                default_tdp=False,
                set_do_not_normalize_lemma_sequence_scores=False,
                set_no_tying_dense=False,
                rtf=2,
            )

            output_time_step = 40 / 1000

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

            tse_job = ComputeTimestampErrorJob(
                allophones=allophones.out_allophone_file,
                alignment=a_job.out_alignment_bundle,
                t_step=output_time_step,
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

            tse_tina_job = ComputeTinaTseJob(
                allophones=allophones.out_allophone_file,
                alignment_bundle=a_job.out_alignment_bundle,
                ref_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
                ref_alignment_bundle=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
                ref_t_step=10 / 1000,
                ss_factor=int(output_time_step / (10 / 1000)),
            )
            tse_tina_job.add_alias(f"tse-tina/{a_name}/tse")
            tk.register_output(f"alignments/{a_name}/statistics/tse-tina", tse_tina_job.out_tse)

            tse_w_job = ComputeSilencePercentageJob(a_job.out_alignment_bundle, allophones.out_allophone_file)
            tk.register_output(f"alignments/{a_name}/statistics/sil", tse_w_job.out_percent_sil)

            s.experiments["fh-fs"]["alignment_job"] = a_job

        if alignment_name == "40ms-FF-v8":
            smbr_epochs = 80
            smbr_keep_epochs = [int(v) for v in np.linspace(10, smbr_epochs, 8)]
            smbr_peak_lr = 5e-6

            smbr_name = f"{ft_name}-smbr:{smbr_epochs}"

            s.set_experiment_dict("fh-smbr", "scratch", "di", postfix_name=smbr_name)

            # Fool RASR into accepting subsampling nets
            sampling_cfg = copy.deepcopy(nn_precomputed_returnn_config)
            sampling_cfg.config["network"]["features_sampled"] = {
                "class": "slice",
                "from": "conv_merged",
                "axis": "F",
                "slice_end": s.initial_nn_args["num_input"],
                "register_as_extern_data": "features_sampled",
            }
            smbr_train_tf_flow = make_precomputed_hybrid_tf_feature_flow(
                tf_graph=compile_tf_graph_from_returnn_config(
                    returnn_config=sampling_cfg, returnn_root=returnn_root, returnn_python_exe=s.returnn_python_exe
                ),
                tf_checkpoint=bw_train_job.out_checkpoints[fine_tune_epochs],
                output_layer_name="features_sampled",
            )
            train_flow = add_tf_flow_to_base_flow(s.feature_flows[train_key]["gt"], smbr_train_tf_flow)
            crp = copy.deepcopy(s.crp[s.crp_names["train"]])
            crp.concurrent = 300
            crp.segment_path = corpus.SegmentCorpusJob(
                s.corpora[s.train_key].corpus_file, crp.concurrent
            ).out_segment_path
            ss_features = FeatureExtractionJob(
                crp=crp,
                feature_flow=train_flow,
                parallel=50,
                port_name_mapping={"features": "ss"},
                rtf=0.5,
            )
            feature_path = rasr.FlagDependentFlowAttribute(
                "cache_mode", {"task_dependent": ss_features.out_feature_path["ss"]}
            )

            returnn_config_smbr = diphone_joint_output.augment_to_joint_diphone_softmax(
                returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
                label_info=s.label_info,
                out_joint_score_layer="output",
                log_softmax=True,
                prepare_for_train=True,
            )
            returnn_config_smbr = seq_disc.augment_for_smbr(
                crp=s.crp[s.crp_names["train"]],
                feature_flow_lattice_generation=feature_flow,
                feature_flow_smbr_training=basic_cache_flow(feature_path),
                feature_scorer_lattice_generation=feature_scorer,
                from_output_layer="output",
                beam_limit=20,
                lm_scale=1.3,
                pron_scale=2.0,
                returnn_config=returnn_config_smbr,
                ce_smoothing=0.0,
                smbr_params=seq_disc.SmbrParameters(
                    num_classes=s.label_info.get_n_of_dense_classes(),
                    num_data_dim=50,
                ),
            )
            lrates = oclr.get_learning_rates(
                lrate=smbr_peak_lr,
                increase=0,
                constLR=math.floor(smbr_epochs * 0.45),
                decay=math.floor(smbr_epochs * 0.45),
                decMinRatio=0.1,
                decMaxRatio=1,
            )
            smbr_update_config = returnn.ReturnnConfig(
                config={
                    "batch_size": 10000,
                    "learning_rates": list(
                        np.concatenate([lrates, np.linspace(min(lrates), 1e-6, smbr_epochs - len(lrates))])
                    ),
                    "preload_from_files": {
                        "existing-model": {
                            "init_for_train": True,
                            "ignore_missing": True,
                            "filename": bw_train_job.out_checkpoints[fine_tune_epochs],
                        }
                    },
                    "extern_data": {"data": {"dim": 50}},
                },
                post_config={"cleanup_old_models": {"keep_best_n": 3, "keep": smbr_keep_epochs}},
                python_epilog={
                    "dynamic_lr_reset": "dynamic_learning_rate = None",
                },
            )
            returnn_config_smbr.update(smbr_update_config)

            s.set_returnn_config_for_experiment("fh-smbr", copy.deepcopy(returnn_config_smbr))
            s.set_graph_for_experiment("fh-smbr", override_cfg=nn_precomputed_returnn_config)

            train_args = {
                **s.initial_train_args,
                "cpu_rqmt": 2,
                "mem_rqmt": 24,
                "log_verbosity": 4,
                "num_epochs": smbr_epochs,
                "partition_epochs": partition_epochs,
                "returnn_config": copy.deepcopy(returnn_config_smbr),
            }
            s.returnn_rasr_training(
                experiment_key="fh-smbr",
                train_corpus_key=s.crp_names["train"],
                dev_corpus_key=s.crp_names["cvtrain"],
                nn_train_args=train_args,
                include_alignment=False,
                on_2080=True,
            )

            for ep, crp_k in itertools.product(smbr_keep_epochs, ["dev-other"]):
                s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

                s.set_mono_priors_returnn_rasr(
                    key="fh-smbr",
                    epoch=min(ep, smbr_keep_epochs[-2]),
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

                base_params = s.get_cart_params(key="fh-smbr")
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
                        key="fh-smbr",
                        epoch=ep,
                        crp_corpus=crp_k,
                        n_cart_out=diphone_li.get_n_of_dense_classes(),
                        cart_tree_or_tying_config=tying_cfg,
                        params=cfg,
                        log_softmax_returnn_config=nn_precomputed_returnn_config,
                        calculate_statistics=True,
                        opt_lm_am_scale=True,
                        prior_epoch=min(ep, smbr_keep_epochs[-2]),
                        rtf=8,
                        cpu_rqmt=2,
                        mem_rqmt=4,
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
