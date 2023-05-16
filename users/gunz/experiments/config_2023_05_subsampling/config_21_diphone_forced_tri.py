__all__ = ["run", "run_single"]

import copy
import dataclasses
import random
from dataclasses import dataclass
import itertools

import numpy as np
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import lexicon, rasr, returnn, text

import i6_experiments.common.setups.rasr.util as rasr_util

from ...setups.common.hdf import RasrForcedTriphoneAlignmentToHDF, RasrFeaturesToHdf
from ...setups.common.nn import oclr, returnn_time_tag
from ...setups.common.nn.cache_epilog import hdf_dataset_cache_epilog
from ...setups.common.nn.chunking import subsample_chunking
from ...setups.common.nn.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)
from ...setups.fh import system as fh_system
from ...setups.fh.decoder.config import PriorInfo
from ...setups.fh.network import conformer
from ...setups.fh.factored import PhoneticContext
from ...setups.fh.network import aux_loss, extern_data
from ...setups.fh.network.augment import (
    SubsamplingInfo,
    augment_net_with_diphone_outputs,
    augment_net_with_monophone_outputs,
    augment_net_with_label_pops,
)
from ...setups.fh.priors import combine_priors_across_hmm_states
from ...setups.ls import gmm_args as gmm_setups, rasr_args as lbs_data_setups

from .config import (
    CONF_CHUNKING,
    CONF_FH_DECODING_TENSOR_CONFIG,
    CONF_FOCAL_LOSS,
    CONF_LABEL_SMOOTHING,
    CONF_SA_CONFIG,
    FROM_SCRATCH_CV_INFO,
    L2,
    RASR_ROOT_FH_GUNZ,
    RASR_ROOT_RS_RASR_GUNZ,
    RETURNN_PYTHON_TF15,
    TEST_EPOCH,
    ZHOU_ALLOPHONES,
    ZHOU_SUBSAMPLED_ALIGNMENT,
)

RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_FH_GUNZ, "arch", gs.RASR_ARCH))
RASR_BINARY_PATH.hash_override = "FH_RASR_PATH"
RASR_BINARY_PATH.hash_override = "RS_RASR_PATH"

RS_RASR_BINARY_PATH = tk.Path(os.path.join(RASR_ROOT_RS_RASR_GUNZ, "arch", gs.RASR_ARCH))

RETURNN_PYTHON_EXE = tk.Path(RETURNN_PYTHON_TF15)
RETURNN_PYTHON_EXE.hash_override = "FH_RETURNN_PYTHON_EXE"

train_key = "train-other-960"


@dataclass(frozen=True)
class Experiment:
    alignment: tk.Path
    alignment_name: str
    lr: str
    dc_detection: bool
    run_performance_study: bool
    subsampling_factor: int
    tune_decoding: bool

    focal_loss: float = CONF_FOCAL_LOSS


def run(returnn_root: tk.Path):
    # ******************** Settings ********************

    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    scratch_align = tk.Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True)

    configs = [
        Experiment(
            alignment=scratch_align,
            alignment_name="scratch",
            dc_detection=True,
            lr="v7",
            run_performance_study=False,
            subsampling_factor=4,
            tune_decoding=False,
        ),
    ]
    for exp in configs:
        run_single(
            alignment=exp.alignment,
            alignment_name=exp.alignment_name,
            dc_detection=exp.dc_detection,
            focal_loss=exp.focal_loss,
            returnn_root=returnn_root,
            run_performance_study=exp.run_performance_study,
            subsampling_factor=exp.subsampling_factor,
            tune_decoding=exp.tune_decoding,
            lr=exp.lr,
        )


def run_single(
    *,
    alignment: tk.Path,
    alignment_name: str,
    dc_detection: bool,
    focal_loss: float,
    lr: str,
    returnn_root: tk.Path,
    run_performance_study: bool,
    subsampling_factor: int,
    tune_decoding: bool,
    conf_model_dim: int = 512,
    num_epochs: int = 600,
) -> fh_system.FactoredHybridSystem:
    # ******************** HY Init ********************

    name = f"conf-lr:{lr}-ss:{subsampling_factor}"
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
    if alignment_name == "scratch":
        s.cv_info = FROM_SCRATCH_CV_INFO
    s.base_allophones = ZHOU_ALLOPHONES
    s.label_info = dataclasses.replace(s.label_info, n_states_per_phone=1)

    s.run(steps)

    # *********** Preparation of data input for rasr-returnn training *****************
    s.alignments[train_key] = alignment
    steps_input = rasr_util.RasrSteps()
    steps_input.add_step("extract", rasr_init_args.feature_extraction_args)
    steps_input.add_step("input", data_preparation_args)
    s.run(steps_input)

    s.set_crp_pairings()
    s.set_rasr_returnn_input_datas(
        is_cv_separate_from_train=alignment_name == "scratch",
        input_key="data_preparation",
        chunk_size=CONF_CHUNKING,
    )
    s._update_am_setting_for_all_crps(
        train_tdp_type="default",
        eval_tdp_type="default",
        add_base_allophones=True,
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
        upsample_by_transposed_conv=False,
        feature_stacking_size=subsampling_factor,
    )
    network = network_builder.network
    network = augment_net_with_label_pops(
        network,
        label_info=s.label_info,
        classes_subsampling_info=SubsamplingInfo(factor=subsampling_factor, time_tag_name=time_tag_name),
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
        upsampling=False,
    )

    tying_crp = s.train_input_data[s.crp_names["train"]].get_crp()
    tying_crp.acoustic_model_config.allophones.add_all = True
    tying = lexicon.DumpStateTyingJob(tying_crp).out_state_tying
    with open(FROM_SCRATCH_CV_INFO["features_tkpath_train"], "rt") as feature_bundle:
        feature_caches = [tk.Path(line.strip()) for line in feature_bundle.readlines()]

    alignment = RasrForcedTriphoneAlignmentToHDF(
        alignment_bundle=alignment,
        allophones=tk.Path(ZHOU_ALLOPHONES),
        num_tied_classes=s.label_info.get_n_of_dense_classes(),
        state_tying=tying,
    )
    features = RasrFeaturesToHdf(feature_caches=feature_caches)

    rng = random.Random(1337)
    dev_i = rng.randrange(0, len(features.out_hdf_files))

    train_hdfs = copy.copy(features.out_hdf_files)
    dev_hdf = train_hdfs.pop(dev_i)

    seqs = copy.copy(features.out_single_segment_files)
    dev_seq = seqs.pop(dev_i)

    feature_seq = text.ConcatenateJob(seqs, out_name="segments", zip_out=True).out

    dataset_cfg = {
        "class": "MetaDataset",
        "data_map": {
            "data": ("audio", "features"),
            "classes": ("alignment", "classes"),
        },
        "seq_ordering": f"random:133769420",
    }
    alignment_hdf_config = {
        "class": "NextGenHDFDataset",
        "input_stream_name": "classes",
        "files": [alignment.out_hdf_file],
        "use_lazy_data_integrity_checks": True,
    }
    features_hdf_config = {
        "class": "NextGenHDFDataset",
        "input_stream_name": "features",
        "use_lazy_data_integrity_checks": True,
    }

    base_config = {
        **s.initial_nn_args,
        **oclr.get_oclr_config(num_epochs=num_epochs, schedule=lr),
        **CONF_SA_CONFIG,
        "batch_size": 11000 if lr == "v7" else 6144,
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "chunking": subsample_chunking(CONF_CHUNKING, subsampling_factor),
        "optimizer": {"class": "nadam"},
        "optimizer_epsilon": 1e-8,
        "gradient_noise": 0.0,
        "network": network,
        "extern_data": {
            "data": {"dim": 50, "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)}},
            **extern_data.get_extern_data_config(label_info=s.label_info, time_tag_name=None),
        },
        "train": {
            **dataset_cfg,
            "datasets": {
                "audio": {**features_hdf_config, "files": train_hdfs},
                "alignment": alignment_hdf_config,
            },
            "partition_epoch": partition_epochs["train"],
            "seq_list_file": feature_seq,
        },
        "dev": {
            **dataset_cfg,
            "datasets": {
                "audio": {**features_hdf_config, "files": [dev_hdf]},
                "alignment": alignment_hdf_config,
            },
            "partition_epoch": partition_epochs["dev"],
            "seq_list_file": dev_seq,
        },
    }
    keep_epochs = [TEST_EPOCH, 550, num_epochs]
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
            "cache": hdf_dataset_cache_epilog,
        },
    )

    s.set_experiment_dict("fh", alignment_name, "di", postfix_name=name)
    s.set_returnn_config_for_experiment("fh", copy.deepcopy(returnn_config))

    train_args = {
        **s.initial_train_args,
        "num_epochs": num_epochs,
    }
    s.returnn_training(experiment_key="fh", returnn_config=returnn_config, nn_train_args=train_args, on_2080=False)

    s.set_graph_for_experiment("fh")
    s.experiments["fh"]["priors"] = combine_priors_across_hmm_states(
        PriorInfo.from_diphone_job(
            tk.Path(
                "/work/asr3/raissi/shared_workspaces/gunz/kept-experiments/2023-04--thesis-baselines/priors/di-from-scratch-conf-ph-2-dim-512-ep-600-cls-WE-lr-v7-sa-v1-bs-11000-ep-550"
            )
        ),
        dataclasses.replace(s.label_info, n_states_per_phone=3),
        s.label_info,
    )

    s.set_binaries_for_crp("dev-other", RS_RASR_BINARY_PATH)
    recognizer, recog_args = s.get_recognizer_and_args(
        key="fh",
        context_type=PhoneticContext.diphone,
        crp_corpus="dev-other",
        epoch=TEST_EPOCH,
        gpu=False,
        tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
        recompile_graph_for_feature_scorer=True,
    )
    recognizer.recognize_count_lm(
        label_info=s.label_info,
        search_parameters=recog_args,
        num_encoder_output=conf_model_dim,
        rerun_after_opt_lm=True,
        calculate_stats=True,
    )

    for ep, crp_k in itertools.product([max(keep_epochs)], ["dev-other"]):
        s.set_binaries_for_crp(crp_k, RS_RASR_BINARY_PATH)

        recognizer, recog_args = s.get_recognizer_and_args(
            key="fh",
            context_type=PhoneticContext.diphone,
            crp_corpus=crp_k,
            epoch=ep,
            gpu=False,
            tensor_map=CONF_FH_DECODING_TENSOR_CONFIG,
            recompile_graph_for_feature_scorer=True,
        )
        for cfg in [recog_args, recog_args.with_prior_scale(0.2, 0.1).with_tdp_scale(0.4)]:
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=cfg,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
            )

        if tune_decoding:
            best_config = recognizer.recognize_optimize_scales(
                label_info=s.label_info,
                search_parameters=recog_args,
                num_encoder_output=conf_model_dim,
                prior_scales=list(
                    itertools.product(
                        np.linspace(0.1, 0.5, 5),
                        np.linspace(0.0, 0.4, 5),
                    )
                ),
                tdp_scales=np.linspace(0.2, 0.6, 5),
            )
            recognizer.recognize_count_lm(
                label_info=s.label_info,
                search_parameters=best_config,
                num_encoder_output=conf_model_dim,
                rerun_after_opt_lm=True,
                calculate_stats=True,
                name_override="best/4gram",
            )

    return s
