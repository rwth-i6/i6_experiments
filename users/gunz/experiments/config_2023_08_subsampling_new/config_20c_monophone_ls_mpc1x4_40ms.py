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

from i6_core import lexicon, mm, rasr, returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.nn import oclr, returnn_time_tag
from ...setups.common.nn.chunking import subsample_chunking
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext, RasrStateTying
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    SubsamplingInfo,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.fh.priors import get_mono_transcription_priors
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
    decode_all_corpora: bool
    lr: str
    multitask: bool
    dc_detection: bool
    run_performance_study: bool
    tune_decoding: bool
    run_tdp_study: bool

    filter_segments: typing.Optional[typing.List[str]] = None
    focal_loss: float = CONF_FOCAL_LOSS


def run(
    returnn_root: tk.Path, alignment: tk.Path, a_name: str
) -> typing.Tuple[Experiment, fh_system.FactoredHybridSystem]:
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    exp = Experiment(
        alignment=alignment,
        alignment_name=a_name,
        dc_detection=False,
        decode_all_corpora=False,
        lr="v13",
        multitask=True,
        run_performance_study=a_name == "40ms-FF-v8",
        tune_decoding=True,
        run_tdp_study=False,
    )
    sys = run_single(
        alignment=exp.alignment,
        alignment_name=exp.alignment_name,
        dc_detection=exp.dc_detection,
        decode_all_corpora=exp.decode_all_corpora,
        focal_loss=exp.focal_loss,
        lr=exp.lr,
        multitask=exp.multitask,
        returnn_root=returnn_root,
        run_performance_study=exp.run_performance_study,
        tune_decoding=exp.tune_decoding,
        filter_segments=exp.filter_segments,
        run_tdp_study=exp.run_tdp_study,
    )
    return exp, sys


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    dc_detection: bool,
    decode_all_corpora: bool,
    focal_loss: float,
    lr: str,
    multitask: bool,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    filter_segments: typing.Optional[typing.List[str]],
    run_tdp_study: bool,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-1-a:{alignment_name}-lr:{lr}-fl:{focal_loss}-mt:{int(multitask)}"
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
        chunk_size=CONF_CHUNKING_10MS,
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
        label_smoothing=0.1,
        use_multi_task=multitask,
    )
    network = aux_loss.add_intermediate_loss(
        network,
        center_state_only=True,
        context=PhoneticContext.monophone,
        encoder_output_len=conf_model_dim,
        focal_loss_factor=focal_loss,
        l2=L2,
        label_info=s.label_info,
        label_smoothing=0.0,
        time_tag_name=time_tag_name,
        upsampling=False,
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
        "chunking": subsample_chunking(CONF_CHUNKING_10MS, ss_factor),
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

    s.set_experiment_dict("fh", alignment_name, "mono", postfix_name=name)
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

        if ep <= 300:
            s.experiments["fh"]["priors"] = get_mono_transcription_priors(1, True)
            s.set_graph_for_experiment("fh")
        elif ep <= keep_epochs[-2]:
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
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
            lm_gc_simple_hash=True,
        )
        recog_args = recog_args.with_lm_scale(round(recog_args.lm_scale / float(ss_factor), 2)).with_prior_scale(0.6)

        # Top 3 from monophone TDP study
        good_values = [
            (0.4, (3, 0, "infinity", 0), (3, 10, "infinity", 10)),  # 8,9%
            (0.6, (3, 0, "infinity", 3), (3, 10, "infinity", 10)),  # 8,9%
            (0.2, (3, 0, "infinity", 0), (10, 10, "infinity", 10)),  # 9,0%
        ]

        for cfg in [
            recog_args.with_tdp_scale(0.1),
            recog_args.with_tdp_scale(0.2),
            recog_args.with_tdp_scale(0.4),
            *(
                recog_args.with_tdp_scale(sc).with_tdp_speech(tdp_sp).with_tdp_silence(tdp_sil)
                for sc, tdp_sp, tdp_sil in good_values
            ),
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
                tdp_speech=[(3, 0, "infinity", 0)],
                tdp_sil=[(3, 10, "infinity", 10), (0, 3, "infinity", 20)],
                prior_scales=np.linspace(0.0, 0.8, 9),
                tdp_scales=[0.2, 0.4, 0.6] if "FF" in alignment_name else [0.4],
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
        nn_precomputed_returnn_config = copy.deepcopy(returnn_config)
        nn_precomputed_returnn_config.config["network"]["center-output"] = {
            **nn_precomputed_returnn_config.config["network"]["center-output"],
            "class": "linear",
            "activation": "log_softmax",
            "register_as_extern_data": "center-output",
        }

        monophone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.monophone)
        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "monophone-dense"

        good_values = [
            (0.4, (3, 0, "infinity", 0), (3, 0, "infinity", 20)),  # default
            (0.4, (3, 0, "infinity", 0), (3, 10, "infinity", 10)),  # 8,9%
            (0.6, (3, 0, "infinity", 3), (3, 10, "infinity", 10)),  # 8,9%
            (0.2, (3, 0, "infinity", 0), (10, 10, "infinity", 10)),  # 9,0%
        ]
        configs = [
            dataclasses.replace(
                s.get_cart_params("fh"),
                altas=a,
                beam=beam,
                beam_limit=100000,
                lm_scale=1.5,
                tdp_scale=tdpSc,
                tdp_silence=tdpSil,
                tdp_speech=tdpSp,
            ).with_prior_scale(pC)
            for beam, a, pC, (tdpSc, tdpSp, tdpSil) in itertools.product(
                [18, 20, 22], [None, 2, 4, 6], [0.2, 0.4, 0.6], good_values
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
                encoder_output_layer="center__output",
                mem_rqmt=4,
                n_cart_out=monophone_li.get_n_of_dense_classes(),
                params=cfg,
                rtf=1.5,
            )
            # j.rqmt.update({"sbatch_args": ["-p", "rescale_amd"]})

    if alignment_name == "40ms-FF-v8":
        nn_precomputed_returnn_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)
        nn_precomputed_returnn_config.config["network"] = {
            **nn_precomputed_returnn_config.config["network"],
            "center-output-expanded": {
                "class": "expand_dims",
                "from": "center-output",
                "axis": "F",
                "dim": 42,
            },
            "center-output-merged": {
                "class": "merge_dims",
                "from": "center-output-expanded",
                "axes": ["dim:84", "dim:42"],
                "keep_order": True,
            },
            "center-output-normalized": {
                "class": "eval",
                "eval": f"tf.math.divide(source(0), 42)",
                "from": "center-output-merged",
            },
            "output": {
                "class": "eval",
                "eval": "tf.math.log(source(0))",
                "from": "center-output-normalized",
                "register_as_extern_data": "output",
            },
        }

        prior_config = copy.deepcopy(nn_precomputed_returnn_config)
        prior_config.config["network"]["output"] = {
            "class": "copy",
            "from": "center-output-normalized",
            "register_as_extern_data": "output",
        }
        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=keep_epochs[-2],
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=prior_config,
            output_layer_name="output",
        )

        diphone_li = dataclasses.replace(s.label_info, state_tying=RasrStateTying.diphone)
        tying_cfg = rasr.RasrConfig()
        tying_cfg.type = "diphone-dense"

        configs = [
            dataclasses.replace(
                s.get_cart_params("fh"),
                altas=a,
                beam=beam,
                beam_limit=100000,
                lm_scale=1.5,
                tdp_scale=tdpSc,
            ).with_prior_scale(pC)
            for beam, a, pC, tdpSc in itertools.product([18, 20, 22], [None, 2, 4, 6], [0.2, 0.4, 0.6], [0.4, 0.6])
        ]
        for cfg in configs:
            s.recognize_cart(
                key="fh",
                epoch=max(keep_epochs),
                calculate_statistics=True,
                cart_tree_or_tying_config=tying_cfg,
                cpu_rqmt=2,
                crp_corpus="dev-other",
                lm_gc_simple_hash=True,
                log_softmax_returnn_config=nn_precomputed_returnn_config,
                encoder_output_layer="output",
                alias_output_prefix="recog-fake-diphone/",
                mem_rqmt=4,
                n_cart_out=diphone_li.get_n_of_dense_classes(),
                params=cfg,
                rtf=4,
            )

    if run_tdp_study:
        base_config = remove_label_pops_and_losses_from_returnn_config(returnn_config)

        s.set_mono_priors_returnn_rasr(
            key="fh",
            epoch=max(keep_epochs),
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=base_config,
        )

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.monophone,
            crp_corpus="dev-other",
            epoch=max(keep_epochs),
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
            lm_gc_simple_hash=True,
        )

        search_cfg = recog_args.with_prior_scale(0.6)
        tdps = itertools.product(
            [0, 3, 10],
            [0],
            [0, 3, 10],
            [0, 3, 10],
            [10],
            [1, 3, 10],
            (0.1, *((round(v, 1) for v in np.linspace(0.2, 0.6, 3)))),
        )
        run_jobs = {}
        for cfg in tdps:
            sp_loop, sp_fwd, sp_exit, sil_loop, sil_fwd, sil_exit, tdp_scale = cfg
            sp_tdp = (sp_loop, sp_fwd, "infinity", sp_exit)
            sil_tdp = (sil_loop, sil_fwd, "infinity", sil_exit)
            params = dataclasses.replace(
                search_cfg,
                altas=2,
                lm_scale=1.33,
                tdp_speech=sp_tdp,
                tdp_silence=sil_tdp,
                tdp_non_word=sil_tdp,
                tdp_scale=tdp_scale,
            )

            jobs = recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=params,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=False,
                calculate_stats=True,
                cpu_rqmt=2,
                mem_rqmt=4,
                rtf_cpu=4,
            )
            run_jobs[tdp_scale, sp_tdp, sil_tdp] = (jobs, params)

        best_sp = (3, 0, "infinity", 0)
        best_sil = (3, 10, "infinity", 10)
        best, params = run_jobs[0.4, best_sp, best_sil]
        tune_tdp_job = mm.ViterbiTdpTuningJob(
            crp=best.search_crp,
            feature_flow=s.feature_flows["dev-other"],
            feature_scorer=best.search_feature_scorer,
            allophone_files=lexicon.StoreAllophonesJob(best.search_crp).out_allophone_file,
            am_args={"tdp_transition": best_sp, "tdp_silence": best_sil},
        )
        tk.register_output(f"tdp-tuning/{name}/opt", tune_tdp_job.am_args_opt)

    if decode_all_corpora:
        assert False, "this is broken r/n"

        for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-clean", "dev-other", "test-clean", "test-other"]):
            s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

            recognizer, recog_args = s.get_recognizer_and_args(
                key="fh",
                context_type=PhoneticContext.monophone,
                crp_corpus=crp_k,
                epoch=ep,
                gpu=False,
                tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
                set_batch_major_for_feature_scorer=True,
            )

            cfgs = [recog_args.with_prior_scale(0.6).with_tdp_scale(0.5)]

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
                context_type=PhoneticContext.monophone,
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
                    rtf_gpu=18,
                    gpu=True,
                )

    return s
