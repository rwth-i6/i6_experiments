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
from sisyphus.delayed_ops import DelayedFallback

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
from ...setups.common.power_consumption import WritePowerConsumptionScriptJob
from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import PriorInfo
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    augment_net_with_triphone_outputs,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.fh.priors import smoothen_priors
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
    lr: str
    dc_detection: bool
    decode_all_corpora: bool
    n_states_per_phone: int
    own_priors: bool
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
            dc_detection=False,
            decode_all_corpora=False,
            lr="v13",
            n_states_per_phone=3,
            own_priors=False,
            run_performance_study=False,
            tune_decoding=False,
        ),
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            dc_detection=False,
            decode_all_corpora=False,
            lr="v13",
            n_states_per_phone=3,
            own_priors=True,
            run_performance_study=True,
            tune_decoding=True,
        ),
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            dc_detection=False,
            decode_all_corpora=False,
            lr="v13",
            n_states_per_phone=1,
            own_priors=True,
            run_performance_study=True,
            tune_decoding=True,
        ),
        *(
            Experiment(
                alignment=a,
                alignment_name=a_name,
                dc_detection=False,
                decode_all_corpora=False,
                lr="v13",
                n_states_per_phone=3,
                own_priors=True,
                run_performance_study=False,
                tune_decoding=True,
            )
            for a, a_name in (additional_alignments or [])
        )
        # Experiment(
        #     alignment=scratch_align_daniel,
        #     alignment_name="scratch_daniel",
        #     dc_detection=True,
        #     decode_all_corpora=False,
        #     lr="v13",
        #     own_priors=False,
        #     run_performance_study=False,
        #     tune_decoding=False,
        # ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            dc_detection=exp.dc_detection,
            decode_all_corpora=exp.decode_all_corpora,
            focal_loss=exp.focal_loss,
            lr=exp.lr,
            n_states_per_phone=exp.n_states_per_phone,
            own_priors=exp.own_priors,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            tune_decoding=exp.tune_decoding,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    dc_detection: bool,
    decode_all_corpora: bool,
    focal_loss: float,
    lr: str,
    n_states_per_phone: int,
    own_priors: bool,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-3-a:{alignment_name}-lr:{lr}-fl:{focal_loss}-n:{n_states_per_phone}"
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
    network = augment_net_with_triphone_outputs(
        network,
        l2=L2,
        ph_emb_size=s.label_info.ph_emb_size,
        st_emb_size=s.label_info.st_emb_size,
        variant=PhoneticContext.triphone_forward,
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
        "batch_size": 12500,
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
    keep_epochs = [300, 500, 550, num_epochs]
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

    s.set_experiment_dict("fh", alignment_name, "tri", postfix_name=name)
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
        on_2080=True,
    )

    if own_priors or n_states_per_phone != 3:
        s.set_triphone_priors_returnn_rasr(
            key="fh",
            epoch=keep_epochs[-2],
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
        )
    else:
        s.set_graph_for_experiment("fh")
        prior_info = PriorInfo.from_triphone_job(
            "/u/mgunz/gunz/kept-experiments/2023-02--from-scratch-daniel/priors/tri-from-scratch-conf-ph-3-dim-512-ep-60-cls-WE-lr-v6-sa-v1-bs-6144-epoch-575"
            if alignment_name == "scratch_daniel"
            else "/u/mgunz/gunz/kept-experiments/2022-07--baselines/priors/tri-from-GMMtri-conf-ph-3-dim-512-ep-600-cls-WE-lr-v6-sa-v1-bs-6144-fls-False-rp-epoch-550"
        )
        s.experiments["fh"]["priors"] = smoothen_priors(prior_info)

    best_config = None
    eps = [500, max(keep_epochs)] if n_states_per_phone == 1 or "FF" in alignment_name else [max(keep_epochs)]
    for ep, crp_k in itertools.product(eps, ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        if own_priors or n_states_per_phone != 3:
            s.set_triphone_priors_returnn_rasr(
                key="fh",
                epoch=min(ep, keep_epochs[-2]),
                train_corpus_key=s.crp_names["train"],
                dev_corpus_key=s.crp_names["cvtrain"],
                smoothen=True,
                returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
            )
        else:
            s.set_graph_for_experiment("fh")
            prior_info = PriorInfo.from_triphone_job(
                "/u/mgunz/gunz/kept-experiments/2023-02--from-scratch-daniel/priors/tri-from-scratch-conf-ph-3-dim-512-ep-60-cls-WE-lr-v6-sa-v1-bs-6144-epoch-575"
                if alignment_name == "scratch_daniel"
                else "/u/mgunz/gunz/kept-experiments/2022-07--baselines/priors/tri-from-GMMtri-conf-ph-3-dim-512-ep-600-cls-WE-lr-v6-sa-v1-bs-6144-fls-False-rp-epoch-550"
            )
            s.experiments["fh"]["priors"] = smoothen_priors(prior_info)

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.triphone_forward,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
        )

        if n_states_per_phone == 1:
            tdp_sp = recog_args.tdp_speech
            recog_args = recog_args.with_tdp_speech((0, *tdp_sp[1:]))

        for cfg in [recog_args, recog_args.with_prior_scale(0.4, 0.4, 0.2).with_tdp_scale(0.6)]:
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                rtf_cpu=80,
            )

        if ep == max(keep_epochs) and tune_decoding:
            best_config = recognizer.recognize_optimize_scales(
                label_info=s.label_info,
                search_parameters=recog_args,
                num_encoder_output=conf_model_dim,
                prior_scales=list(
                    itertools.product(
                        np.linspace(0.1, 0.5, 5),
                        np.linspace(0.0, 0.4, 3),
                        np.linspace(0.0, 0.2, 3),
                    )
                ),
                tdp_scales=np.linspace(0.2, 0.6, 3),
            )
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=best_config,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                name_override="best/4gram",
                rtf_cpu=80,
            )

            if alignment_name == "scratch" and ep == max(keep_epochs):
                base_cfg = best_config.with_beam_limit(100_000)
                base_cfgs = [
                    ("base", base_cfg),
                    ("all-zero", base_cfg.with_prior_scale(center=0, left=0, right=0)),
                    (
                        "only-center",
                        base_cfg.with_prior_scale(center=base_cfg.prior_info.center_state_prior.scale, left=0, right=0),
                    ),
                    (
                        "only-left-right",
                        base_cfg.with_prior_scale(
                            center=0,
                            left=base_cfg.prior_info.left_context_prior.scale,
                            right=base_cfg.prior_info.right_context_prior.scale,
                        ),
                    ),
                    (
                        "only-center-right",
                        base_cfg.with_prior_scale(
                            left=0,
                            center=base_cfg.prior_info.center_state_prior.scale,
                            right=base_cfg.prior_info.right_context_prior.scale,
                        ),
                    ),
                    (
                        "only-center-left",
                        base_cfg.with_prior_scale(
                            right=0,
                            center=base_cfg.prior_info.center_state_prior.scale,
                            left=base_cfg.prior_info.left_context_prior.scale,
                        ),
                    ),
                ]
                cfgs = [
                    *base_cfgs,
                    *((f"{name}-tdpScale0", cfg.with_tdp_scale(0)) for name, cfg in base_cfgs),
                    *(
                        (
                            f"{name}-tdpZero",
                            cfg.with_tdp_speech((0, 0, "infinity", 0)).with_tdp_silence(
                                (0, 0, "infinity", cfg.tdp_silence[-1])
                            ),
                        )
                        for name, cfg in base_cfgs
                    ),
                ]

                for name, cfg in cfgs:
                    jobs = recognizer.recognize_count_lm(
                        label_info=s.label_info,
                        search_parameters=cfg,
                        num_encoder_output=conf_model_dim,
                        rerun_after_opt_lm=False,
                        calculate_stats=True,
                        name_override=f"icassp/4gram/{name}",
                        rtf_cpu=80,
                    )

            if run_performance_study:
                lm_scale = 10.85 if n_states_per_phone == 3 else 7.85

                power_consumption_script = WritePowerConsumptionScriptJob(s.crp["dev-other"].flf_tool_exe)

                def set_power_exe(crp):
                    crp.flf_tool_exe = power_consumption_script.out_script

                for altas, beam, b_l in itertools.product(
                    [None, 2, 4, 6],
                    [12, 14, 16],
                    [
                        int(v)
                        for v in (
                            *np.geomspace(250, 1000, 4, dtype=int)[:-1],
                            *np.geomspace(1000, max_bl, 10, dtype=int)[:7],
                        )
                    ],
                ):
                    nice = (
                        f"--nice={100 + int(2 * (20 - np.log(b_l)))}"
                        if b_l < 100_000
                        else "--nice=500"
                        if n_states_per_phone < 3
                        else "--nice=100"
                    )

                    jobs = recognizer.recognize_count_lm(
                        calculate_stats=True,
                        gpu=False,
                        label_info=s.label_info,
                        name_override=f"altas{altas}-beam{beam}-beamlimit{b_l}",
                        num_encoder_output=conf_model_dim,
                        opt_lm_am=False,
                        pre_path="decoding-perf-single-core",
                        search_parameters=dataclasses.replace(
                            best_config,
                            altas=altas,
                            beam=beam,
                            beam_limit=b_l,
                            lm_scale=lm_scale,
                        ),
                        cpu_rqmt=2,
                        mem_rqmt=4,
                        rtf_cpu=8 if altas is not None else 12,
                        crp_update=set_power_exe,
                        remove_or_set_concurrency=12,
                    )
                    jobs.search.rqmt.update(
                        {"sbatch_args": [v for v in ["-A", "rescale_speed", "-p", "rescale_amd", nice] if v]}
                    )

                    continue
                    jobs = recognizer.recognize_ls_trafo_lm(
                        calculate_stats=True,
                        gpu=False,
                        label_info=s.label_info,
                        name_override=f"altas{altas}-beam{beam}",
                        num_encoder_output=conf_model_dim,
                        opt_lm_am=False,
                        search_parameters=dataclasses.replace(
                            best_config,
                            altas=altas,
                            beam=beam,
                            beam_limit=100_000,
                            lm_scale=lm_scale + 2,
                        ),
                        cpu_rqmt=2,
                        mem_rqmt=4,
                        rtf_cpu=12 if altas is not None else 20,
                        crp_update=set_power_exe,
                        remove_or_set_concurrency=12,
                    )
                    jobs.search.rqmt.update(
                        {"sbatch_args": [v for v in ["-A", "rescale_speed", "-p", "rescale_amd", nice] if v]}
                    )

    if decode_all_corpora:
        for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-clean", "dev-other", "test-clean", "test-other"]):
            s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

            recognizer, recog_args = s.get_recognizer_and_args(
                key="fh",
                context_type=PhoneticContext.triphone_forward,
                crp_corpus=crp_k,
                epoch=ep,
                gpu=False,
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
            )

            if n_states_per_phone == 1:
                tdp_sp = recog_args.tdp_speech
                recog_args = recog_args.with_tdp_speech((0, *tdp_sp[1:]))

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
                context_type=PhoneticContext.triphone_forward,
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
                    rtf_gpu=24,
                    gpu=True,
                )

    return s
