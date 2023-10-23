__all__ = ["run", "run_single"]

import copy
import dataclasses
import math
import random
import typing
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import Path, gs, tk

# -------------------- Recipes --------------------

from i6_core import lexicon, rasr, returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.hdf.random_allophones import RasrFeatureAndAlignmentWithRandomAllophonesToHDF
from ...setups.common.nn import oclr, returnn_time_tag
from ...setups.common.nn.chunking import subsample_chunking
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import multistage, system as fh_system
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    SubsamplingInfo,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
    augment_net_with_triphone_outputs,
    remove_label_pops_and_losses_from_returnn_config,
)
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CART_TREE_TRI,
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
    lr: str
    dc_detection: bool
    decode_all_corpora: bool
    init_from_system: fh_system.FactoredHybridSystem
    own_priors: bool
    run_performance_study: bool
    tune_decoding: bool

    filter_segments: typing.Optional[typing.List[str]] = None
    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path, alignment: tk.Path, a_name: str, init_from_system: fh_system.FactoredHybridSystem):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    configs = [
        Experiment(
            alignment=alignment,
            alignment_name=a_name,
            batch_size=12500,
            dc_detection=False,
            decode_all_corpora=False,
            init_from_system=init_from_system,
            lr="v13",
            own_priors=False,
            run_performance_study=False,
            tune_decoding=a_name == "40ms-FF-v8",
        ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            batch_size=exp.batch_size,
            dc_detection=exp.dc_detection,
            decode_all_corpora=exp.decode_all_corpora,
            focal_loss=exp.focal_loss,
            init_from_system=exp.init_from_system,
            lr=exp.lr,
            own_priors=exp.own_priors,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            filter_segments=exp.filter_segments,
            tune_decoding=exp.tune_decoding,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    batch_size: int,
    dc_detection: bool,
    decode_all_corpora: bool,
    focal_loss: float,
    init_from_system: fh_system.FactoredHybridSystem,
    lr: str,
    own_priors: bool,
    returnn_root: tk.Path,
    run_performance_study: bool,
    tune_decoding: bool,
    filter_segments: typing.Optional[typing.List[str]],
    conf_model_dim: int = 512,
    num_epochs: int = 450,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-3-a:{alignment_name}-lr:{lr}-bs:{batch_size}-fl:{focal_loss}"
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
        upsampling=False,
    )

    lrates = oclr.get_learning_rates(
        lrate=8e-5,
        increase=0,
        constLR=math.floor(num_epochs * 0.45),
        decay=math.floor(num_epochs * 0.45),
        decMinRatio=0.1,
        decMaxRatio=1,
    )
    base_config = {
        **s.initial_nn_args,
        **CONF_SA_CONFIG,
        "learning_rates": list(
            np.concatenate(
                [lrates, np.linspace(min(lrates), 1e-6, num_epochs - len(lrates))],
            )
        ),
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
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=None),
        },
        "dev": {"reduce_target_factor": ss_factor},
        "train": {"reduce_target_factor": ss_factor},
    }
    keep_epochs = [50, 100, 225, 375, num_epochs]
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
    prev_key = "fh-fs" if "fh-fs" in init_from_system.experiments else "fh"
    ckpts = init_from_system.experiments[prev_key]["train_job"].out_checkpoints
    returnn_config = multistage.transform_checkpoint(
        name=name,
        init_new=multistage.Init.glorot_uniform,
        input_returnn_config=init_from_system.experiments[prev_key]["returnn_config"],
        input_label_info=init_from_system.label_info,
        input_model_path=ckpts[max(ckpts)],
        output_returnn_config=returnn_config,
        output_label_info=s.label_info,
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
    )

    s.set_experiment_dict("fh", alignment_name, "tri", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "num_epochs": num_epochs,
        "partition_epochs": partition_epochs,
        "returnn_config": copy.deepcopy(returnn_config),
    }
    viterbi_train_job = s.returnn_rasr_training(
        experiment_key="fh",
        train_corpus_key=s.crp_names["train"],
        dev_corpus_key=s.crp_names["cvtrain"],
        nn_train_args=train_args,
    )

    best_config = None
    for ep, crp_k in itertools.product([225, num_epochs], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RASR_TF_BINARY_PATH)

        s.set_triphone_priors_returnn_rasr(
            key="fh",
            epoch=ep,
            train_corpus_key=s.crp_names["train"],
            dev_corpus_key=s.crp_names["cvtrain"],
            smoothen=True,
            returnn_config=remove_label_pops_and_losses_from_returnn_config(returnn_config),
            data_share=0.1,
        )

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.triphone_forward,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
        )
        recog_args = recog_args.with_lm_scale(round(recog_args.lm_scale / float(ss_factor), 2)).with_tdp_scale(0.1)

        # Top 3 from monophone TDP study
        good_values = [
            ((3, 0, "infinity", 0), (3, 10, "infinity", 10)),  # 8,8%
            ((3, 0, "infinity", 3), (3, 10, "infinity", 10)),  # 8,9%
            ((3, 0, "infinity", 0), (10, 10, "infinity", 10)),  # 9,0%
        ]

        for cfg in [
            recog_args.with_prior_scale(0.4, 0.4, 0.2),
            recog_args.with_tdp_scale(0.4)
            .with_prior_scale(0.3, 0.2, 0.2)
            .with_tdp_speech((3, 0, "infinity", 0))
            .with_tdp_silence((10, 10, "infinity", 10)),
            *(
                recog_args.with_prior_scale(0.4, 0.4, 0.2)
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
                rtf_cpu=35,
            )

        if tune_decoding and ep == keep_epochs[-1]:
            best_config = recognizer.recognize_optimize_scales(
                label_info=s.label_info,
                search_parameters=recog_args,
                num_encoder_output=conf_model_dim,
                tdp_speech=[(3, 0, "infinity", 0)],
                tdp_sil=[(10, 10, "infinity", 10)],
                prior_scales=list(
                    itertools.product(
                        np.linspace(0.1, 0.5, 5),
                        np.linspace(0.0, 0.4, 3),
                        np.linspace(0.0, 0.4, 3),
                    )
                ),
                tdp_scales=[0.4],
            )
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=best_config,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                name_override="best/4gram",
                rtf_cpu=35,
            )

    fine_tune = True
    if fine_tune:
        ft_share = 0.3
        peak_lr = 8e-5
        fine_tune_epochs = 450
        fine_tune_keep_epochs = [25, 50, 100, 225, 400, 450]

        cart_crp = copy.deepcopy(s.crp[s.crp_names["train"]])
        cart_crp.acoustic_model_config.hmm.states_per_phone = 3
        cart_crp.acoustic_model_config.state_tying.type = "cart"
        cart_crp.acoustic_model_config.state_tying.file = Path(CART_TREE_TRI, cached=True)
        cart_tying_job = lexicon.DumpStateTyingJob(crp=cart_crp)

        dense_crp = copy.deepcopy(s.crp[s.crp_names["train"]])
        dense_crp.acoustic_model_config.hmm.states_per_phone = 3
        dense_tying_job = lexicon.DumpStateTyingJob(crp=dense_crp)

        allophones_job = lexicon.StoreAllophonesJob(crp=s.crp[s.crp_names["train"]])

        with open(s.alignments[train_key], "rt") as bundle:
            a_caches = [Path(l.strip()) for l in bundle]

        r = random.Random()
        r.seed(42 - 1)

        randomized_indices = set(r.sample(list(range(len(a_caches))), k=int(len(a_caches) * ft_share)))
        randomized_a_caches = [a_caches[i] for i in randomized_indices]
        randomized_features = [s.feature_caches[train_key]["gt"].hidden_paths[i + 1] for i in randomized_indices]
        randomized_hdfs = RasrFeatureAndAlignmentWithRandomAllophonesToHDF(
            feature_caches=randomized_features,
            alignment_caches=randomized_a_caches,
            label_info=s.label_info,
            allophones=allophones_job.out_allophone_file,
            cart_tying=cart_tying_job.out_state_tying,
            dense_tying=dense_tying_job.out_state_tying,
        )
        normal_indices = set(range(300)) - randomized_indices
        normal_a_caches = [a_caches[i] for i in normal_indices]
        normal_features = [s.feature_caches[train_key]["gt"].hidden_paths[i + 1] for i in normal_indices]
        normal_hdfs = RasrFeatureAndAlignmentWithRandomAllophonesToHDF(
            feature_caches=normal_features,
            alignment_caches=normal_a_caches,
            label_info=s.label_info,
            allophones=allophones_job.out_allophone_file,
            cart_tying=cart_tying_job.out_state_tying,
            dense_tying=dense_tying_job.out_state_tying,
        )
        # do not pop dev_hdf from the job output list but from a copy instead
        normal_hdf_files = list(normal_hdfs.out_hdf_files)
        dev_hdf = normal_hdf_files.pop(-1)

        ft_config = copy.deepcopy(returnn_config)
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
                        "filename": viterbi_train_job.out_checkpoints[num_epochs],
                    }
                },
                "extern_data": {"data": {"dim": 50}},
            },
            post_config={"cleanup_old_models": {"keep_best_n": 3, "keep": fine_tune_keep_epochs}},
            python_epilog={
                "dynamic_lr_reset": "dynamic_learning_rate = None",
            },
        )
        ft_config.update(update_config)

        train_args = {
            **s.initial_train_args,
            "num_epochs": fine_tune_epochs,
            "partition_epochs": partition_epochs,
        }
        s.returnn_training_from_hdf(
            experiment_key="fh",
            returnn_config=returnn_config,
            nn_train_args=train_args,
            train_hdfs=normal_hdf_files + randomized_hdfs.out_hdf_files,
            dev_hdfs=[dev_hdf],
            on_2080=False,
        )

    if run_performance_study:
        assert tune_decoding

        ep = 600
        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.triphone_forward,
            crp_corpus="dev-other",
            epoch=ep,
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            set_batch_major_for_feature_scorer=True,
            lm_gc_simple_hash=True,
        )
        recog_args = dataclasses.replace(best_config, altas=4, beam=14, lm_scale=best_config.lm_scale + 0.01)
        jobs = recognizer.recognize_count_lm(
            label_info=s.label_info,
            search_parameters=recog_args,
            num_encoder_output=conf_model_dim,
            rerun_after_opt_lm=True,
            calculate_stats=True,
            pre_path="decoding-perf-eval",
            name_override="best/4gram",
            cpu_rqmt=2,
            mem_rqmt=4,
        )
        jobs.search.rqmt.update({"sbatch_args": ["-p", "rescale_amd"]})

    if decode_all_corpora:
        assert False, "this is broken r/n"

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
