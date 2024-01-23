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
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.common.power_consumption import WritePowerConsumptionScriptJob
from ...setups.fh import system as fh_system
from ...setups.fh.network import conformer, diphone_joint_output
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING,
    CONF_FH_DECODING_TENSOR_CONFIG,
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
    batch_size: int
    decode_all_corpora: bool
    lr: str
    dc_detection: bool
    n_states_per_phone: int
    run_performance_study: bool
    tune_decoding: bool

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path, additional_alignments: typing.Optional[typing.List[typing.Tuple[tk.Path, str]]] = None):
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
            batch_size=12500,
            dc_detection=False,
            decode_all_corpora=False,
            lr="v13",
            n_states_per_phone=3,
            run_performance_study=False,
            tune_decoding=False,
        ),
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            batch_size=12500,
            dc_detection=False,
            decode_all_corpora=True,
            lr="v13",
            n_states_per_phone=3,
            run_performance_study=True,
            tune_decoding=False,
        ),
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            batch_size=12500,
            dc_detection=False,
            decode_all_corpora=False,
            lr="v13",
            n_states_per_phone=1,
            run_performance_study=True,  # TODO fixme w/ proper params
            tune_decoding=False,
        ),
        *(
            Experiment(
                alignment=a,
                alignment_name=a_name,
                batch_size=12500,
                dc_detection=False,
                decode_all_corpora=False,
                lr="v13",
                n_states_per_phone=3,
                run_performance_study=False,
                tune_decoding=True,
            )
            for a, a_name in (additional_alignments or [])
        )
        # Experiment(
        #     alignment=scratch_align_daniel,
        #     alignment_name="scratch_daniel",
        #     batch_size=12500,
        #     dc_detection=True,
        #     decode_all_corpora=False,
        #     lr="v13",
        #     run_performance_study=False,
        #     tune_decoding=False,
        # ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            batch_size=exp.batch_size,
            dc_detection=exp.dc_detection,
            decode_all_corpora=exp.decode_all_corpora,
            focal_loss=exp.focal_loss,
            n_states_per_phone=exp.n_states_per_phone,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            tune_decoding=exp.tune_decoding,
            lr=exp.lr,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    batch_size: int,
    dc_detection: bool,
    decode_all_corpora: bool,
    focal_loss: float,
    lr: str,
    n_states_per_phone: int,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-2-a:{alignment_name}-lr:{lr}-fl:{focal_loss}-n:{n_states_per_phone}"
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
    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=n_states_per_phone)

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
    network_builder = conformer.get_best_model_config(
        conf_model_dim,
        chunking=CONF_CHUNKING,
        focal_loss_factor=CONF_FOCAL_LOSS,
        label_smoothing=CONF_LABEL_SMOOTHING,
        num_classes=s.label_info.get_n_of_dense_classes(),
        time_tag_name=time_tag_name,
    )
    network = network_builder.network
    network = augment_net_with_label_pops(network, label_info=s.label_info)
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
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=CONF_LABEL_SMOOTHING,
        time_tag_name=time_tag_name,
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
        "chunking": CONF_CHUNKING,
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
    keep_epochs = [500, 550, num_epochs]
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
        "returnn_config": returnn_config,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
    }
    viterbi_train_j = s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
        on_2080=True,
    )

    best_config = None
    eps = (
        keep_epochs
        if "FF" in alignment_name
        else [500, max(keep_epochs)]
        if n_states_per_phone == 1
        else [max(keep_epochs)]
    )
    for ep, crp_k in itertools.product(eps, ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        s.set_diphone_priors_returnn_rasr(
            key="fh",
            epoch=min(ep, keep_epochs[-2]),
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

        if n_states_per_phone == 1:
            tdp_sp = recog_args.tdp_speech
            recog_args = recog_args.with_tdp_speech((0, *tdp_sp[1:]))

        for cfg in [
            recog_args.with_prior_scale(0.4, 0.4).with_tdp_scale(0.4),
            recog_args.with_prior_scale(0.2, 0.1).with_tdp_scale(0.4),
        ]:
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
            )

        if tune_decoding and ep == max(keep_epochs):
            best_config = recognizer.recognize_optimize_scales(
                label_info=s.label_info,
                search_parameters=recog_args,
                num_encoder_output=conf_model_dim,
                prior_scales=list(
                    itertools.product(
                        np.linspace(0.2, 0.8, 4) if "FF" in alignment_name else np.linspace(0.1, 0.7, 7),
                        np.linspace(0.0, 0.8, 5) if "FF" in alignment_name else np.linspace(0.0, 0.5, 6),
                    )
                ),
                tdp_scales=np.linspace(0.2, 0.8, 4) if "FF" in alignment_name else np.linspace(0.2, 0.6, 5),
            )
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=best_config,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                name_override="best/4gram",
            )

    if run_performance_study:
        power_consumption_script = WritePowerConsumptionScriptJob(s.crp["dev-other"].flf_tool_exe)

        def set_power_exe(crp):
            crp.flf_tool_exe = power_consumption_script.out_script

        prior_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=returnn_config, label_info=s.label_info, out_joint_score_layer="output", log_softmax=False
        )
        s.set_mono_priors_returnn_rasr(
            "fh",
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            epoch=keep_epochs[-2],
            output_layer_name="output",
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(
                prior_returnn_config, except_layers=["pastLabel"]
            ),
        )

        nn_precomputed_returnn_config = diphone_joint_output.augment_to_joint_diphone_softmax(
            returnn_config=returnn_config, label_info=s.label_info, out_joint_score_layer="output", log_softmax=True
        )
        s.set_graph_for_experiment("fh", override_cfg=nn_precomputed_returnn_config)

        diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)

        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "diphone-dense"

        max_bl = 10000
        for a, pC, b, b_l in itertools.product(
            [None, 2, 4, 6, 8],
            [0.6],
            [12, 14, 16, 18, 20],
            [int(v) for v in (*np.geomspace(250, 1000, 4, dtype=int)[:-1], *np.geomspace(1000, max_bl, 10, dtype=int))]
            if n_states_per_phone == 3
            else [100_000],
        ):
            cfg = dataclasses.replace(
                s.get_cart_params("fh").with_prior_scale(pC),
                altas=a,
                beam=b,
                beam_limit=b_l,
                lm_scale=7.51,
                tdp_scale=0.4 if n_states_per_phone == 3 else 0.2,
            )
            nice = "--nice=500" if n_states_per_phone < 3 else f"--nice={int(max_bl - b_l / 1000)}"
            s.recognize_cart(
                key="fh",
                epoch=max(keep_epochs),
                crp_corpus="dev-other",
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                cart_tree_or_tying_config=tying_cfg,
                params=cfg,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                calculate_statistics=True,
                opt_lm_am_scale=False,
                cpu_rqmt=2,
                mem_rqmt=4,
                remove_or_set_concurrency=12,
                crp_update=set_power_exe,
                rtf=2,
                search_rqmt_update={
                    "sbatch_args": [v for v in ["-A", "rescale_speed", "-p", "rescale_amd", nice] if v]
                },
            )

    if decode_all_corpora:
        assert run_performance_study

        base_params = dataclasses.replace(
            s.get_cart_params("fh").with_prior_scale(0.6),
            beam=18,
            beam_limit=15_000,
            lm_scale=8.4,
            lm_lookahead_scale=4.2,
            tdp_scale=0.4 if n_states_per_phone == 3 else 0.2,
        )

        for crp_k, ep in itertools.product(
            ["test-clean", "test-other", "dev-other", "dev-clean"],
            [590, 599, max(keep_epochs)],
        ):
            s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

            s.recognize_cart(
                key="fh",
                epoch=ep,
                crp_corpus=crp_k,
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                cart_tree_or_tying_config=tying_cfg,
                params=base_params,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                calculate_statistics=True,
                opt_lm_am_scale=False,
                cpu_rqmt=2,
                mem_rqmt=4,
                crp_update=set_power_exe,
                rtf=2,
            )
            s.recognize_cart(
                key="fh",
                epoch=ep,
                crp_corpus=crp_k,
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                cart_tree_or_tying_config=tying_cfg,
                params=base_params.with_lm_scale(11.4),
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                calculate_statistics=True,
                opt_lm_am_scale=False,
                cpu_rqmt=2,
                mem_rqmt=8,
                rtf=20,
                decode_trafo_lm=True,
                recognize_only_trafo=True,
                remove_or_set_concurrency=5,
                gpu=True,
            )

        kept_epochs = [550, 561, 562, 573, 575, 576, 577, 578, 579, 584, 585, 586, 589, 590, 594, 595, 597, 599, 600]
        for ep in kept_epochs:
            s.recognize_cart(
                key="fh",
                epoch=ep,
                crp_corpus="dev-other",
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                cart_tree_or_tying_config=tying_cfg,
                params=base_params,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                calculate_statistics=True,
                opt_lm_am_scale=False,
                cpu_rqmt=2,
                mem_rqmt=4,
                crp_update=set_power_exe,
                rtf=2,
            )

    # ###########
    # FINE TUNING
    # ###########

    fine_tune = alignment_name == "scratch" or n_states_per_phone == 3
    if fine_tune:
        fine_tune_epochs = 450
        fine_tune_keep_epochs = [15, 25, 225, 400, 450]
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
                    "batch_size": 4096,
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
                    "extern_data": {
                        "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
                    },
                },
                post_config={"cleanup_old_models": {"keep_best_n": 3, "keep": fine_tune_keep_epochs}},
                python_epilog={
                    "dynamic_lr_reset": "dynamic_learning_rate = None",
                },
            )
            returnn_config_ft.update(update_config)

            s.set_returnn_config_for_experiment("fh-fs", copy.deepcopy(returnn_config_ft))

            train_args = {
                **s.initial_train_args,
                "num_epochs": fine_tune_epochs,
                "mem_rqmt": 10,
                "partition_epochs": partition_epochs,
                "returnn_config": copy.deepcopy(returnn_config_ft),
            }
            s.returnn_rasr_training(
                experiment_key="fh-fs",
                train_corpus_key=s.crp_names["train"],
                dev_corpus_key=s.crp_names["cvtrain"],
                nn_train_args=train_args,
            )

            for ep, crp_k in itertools.product(fine_tune_keep_epochs, ["dev-other"]):
                s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

                s.set_mono_priors_returnn_rasr(
                    key="fh-fs",
                    epoch=min(ep, fine_tune_keep_epochs[-2]),
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
                decoding_cfgs = (
                    [
                        dataclasses.replace(base_params, lm_scale=5, tdp_scale=sc).with_prior_scale(pC)
                        for sc, pC in itertools.product([0.2, 0.4, 0.6], [0.2, 0.4, 0.6, 0.8])
                    ]
                    if ep == max(fine_tune_keep_epochs)
                    else [dataclasses.replace(base_params, lm_scale=5, tdp_scale=0.2).with_prior_scale(0.4)]
                )
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
                        rtf=12,
                    )

    integrated_training_schedule = alignment_name in ["scratch", "10ms-FF-v8"] and n_states_per_phone == 3
    if integrated_training_schedule:
        orig_name = name
        for start_ep in [550]:
            fine_tune_epochs = num_epochs - start_ep
            fine_tune_keep_epochs = [int(v) for v in np.linspace(fine_tune_epochs * 0.1, fine_tune_epochs, 4)]

            bw_scale = baum_welch.BwScales(label_posterior_scale=1.0, label_prior_scale=None, transition_scale=0.3)

            fine_tune_name = f"{orig_name}-fs_integrated:{start_ep}-bwl:{bw_scale.label_posterior_scale}-bwt:{bw_scale.transition_scale}"
            s.set_experiment_dict("fh-fs-integrated", alignment_name, "di", postfix_name=fine_tune_name)

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
            update_config = returnn.ReturnnConfig(
                config={
                    "batch_size": 4096,
                    "learning_rates": returnn_config.config["learning_rates"][start_ep:],
                    "preload_from_files": {
                        "existing-model": {
                            "init_for_train": True,
                            "ignore_missing": True,
                            "filename": viterbi_train_j.out_checkpoints[start_ep],
                        }
                    },
                    "extern_data": {
                        "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
                    },
                },
                post_config={"cleanup_old_models": {"keep_best_n": 3, "keep": fine_tune_keep_epochs}},
                python_epilog={
                    "dynamic_lr_reset": "dynamic_learning_rate = None",
                },
            )
            returnn_config_ft.update(update_config)

            s.set_returnn_config_for_experiment("fh-fs-integrated", copy.deepcopy(returnn_config_ft))

            train_args = {
                **s.initial_train_args,
                "num_epochs": fine_tune_epochs,
                "mem_rqmt": 10,
                "partition_epochs": partition_epochs,
                "returnn_config": copy.deepcopy(returnn_config_ft),
            }
            s.returnn_rasr_training(
                experiment_key="fh-fs-integrated",
                train_corpus_key=s.crp_names["train"],
                dev_corpus_key=s.crp_names["cvtrain"],
                nn_train_args=train_args,
            )

            for ep, crp_k in itertools.product(fine_tune_keep_epochs, ["dev-other"]):
                s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

                s.set_mono_priors_returnn_rasr(
                    key="fh-fs-integrated",
                    epoch=ep,
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

                base_params = s.get_cart_params(key="fh-fs-integrated")
                decoding_cfgs = [
                    dataclasses.replace(base_params, lm_scale=5, tdp_scale=sc).with_prior_scale(pC)
                    for sc, pC in [(0.4, 0.3), (0.2, 0.4), (0.4, 0.4), (0.2, 0.5)]
                ]
                for cfg in decoding_cfgs:
                    s.recognize_cart(
                        key="fh-fs-integrated",
                        epoch=ep,
                        crp_corpus=crp_k,
                        n_cart_out=diphone_li.get_n_of_dense_classes(),
                        cart_tree_or_tying_config=tying_cfg,
                        params=cfg,
                        log_softmax_returnn_config=nn_precomputed_returnn_config,
                        calculate_statistics=True,
                        opt_lm_am_scale=True,
                        rtf=12,
                    )

    return s
