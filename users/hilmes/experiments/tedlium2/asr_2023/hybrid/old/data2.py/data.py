from typing import Optional, Dict, Any, Tuple, Callable
from sisyphus import tk

from i6_core import corpus as corpus_recipe
from i6_core.returnn import ReturnnDumpHDFJob
from i6_core.features import FeatureExtractionJob

from i6_experiments.common.datasets.tedlium2.constants import DURATIONS, NUM_SEGMENTS
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.old.data import (
    HdfDataInput,
    AllophoneLabeling,
    ReturnnRasrDataInput,
)
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.old.baseline_args import get_align_dev_args

# from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT
from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.hybrid.default_tools import (
    RETURNN_RC_ROOT,
    RETURNN_EXE,
)
from i6_core.lexicon import DumpStateTyingJob


def build_hdf_data_input(
    features: tk.Path,
    allophone_labeling: AllophoneLabeling,
    alignments: tk.Path,
    segment_list: Optional[tk.Path] = None,
    alias_prefix: Optional[str] = None,
):

    feat_dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": features,
                "data_type": "feat",
                "allophone_labeling": {
                    "silence_phone": allophone_labeling.silence_phoneme,
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            }
        },
        "seq_list_filter_file": segment_list,
    }

    feat_job = ReturnnDumpHDFJob(
        data=feat_dataset,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
    )
    if alias_prefix is not None:
        feat_job.add_alias(alias_prefix + "/dump_features")
    feat_hdf = feat_job.out_hdf
    align_dataset = {
        "data": {
            "data": {
                "filename": alignments,
                "data_type": "align",
                "allophone_labeling": {
                    "silence_phone": allophone_labeling.silence_phoneme,
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            }
        },
        "seq_list_filter_file": segment_list,
        "class": "SprintCacheDataset",
    }
    align_job = ReturnnDumpHDFJob(data=align_dataset, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_RC_ROOT)
    if alias_prefix is not None:
        align_job.add_alias(alias_prefix + "/dump_alignments")
    align_hdf = align_job.out_hdf

    return HdfDataInput(features=feat_hdf, alignments=align_hdf)


def dump_features_for_hybrid_training(
    gmm_system: GmmSystem,
    feature_extraction_args: Dict[str, Any],
    feature_extraction_class: Callable[[Any, ...], FeatureExtractionJob],
) -> Tuple[tk.Path, tk.Path, tk.Path]:
    features = {}
    for name in ["nn-train", "nn-cv", "nn-devtrain"]:
        features[name] = feature_extraction_class(gmm_system.crp[name], **feature_extraction_args).out_feature_bundle

    return features["nn-train"], features["nn-cv"], features["nn-devtrain"]


def get_corpus_data_inputs(
    gmm_system: GmmSystem,
    feature_extraction_args: Dict[str, Any],
    feature_extraction_class: Callable[[Any], FeatureExtractionJob],
    alias_prefix: Optional[str] = None,
) -> Tuple[
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, ReturnnRasrDataInput],
    Dict[str, ReturnnRasrDataInput],
]:

    train_corpus_path = gmm_system.corpora["train"].corpus_file
    cv_corpus_path = gmm_system.corpora["dev"].corpus_file

    total_train_num_segments = NUM_SEGMENTS["train"]
    total_cv_num_segments = NUM_SEGMENTS["dev"]

    all_train_segments = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1).out_single_segment_files[1]
    all_cv_segments = corpus_recipe.SegmentCorpusJob(cv_corpus_path, 1).out_single_segment_files[1]

    dev_train_size = 500 / total_train_num_segments
    cv_size = 150 / total_cv_num_segments

    splitted_train_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_train_segments,
        {"devtrain": dev_train_size, "unused": 1 - dev_train_size},
    )
    splitted_cv_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_cv_segments,
        {"cv": cv_size, "unused": 1 - cv_size},
    )

    devtrain_segments = splitted_train_segments_job.out_segments["devtrain"]
    cv_segments = splitted_cv_segments_job.out_segments["cv"]

    cv_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(cv_corpus_path, cv_segments).out_corpus
    devtrain_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(train_corpus_path, devtrain_segments).out_corpus

    # ******************** NN Init ********************
    gmm_system.add_overlay("train", "nn-train")
    gmm_system.crp["nn-train"].segment_path = all_train_segments
    gmm_system.crp["nn-train"].concurrent = 1

    gmm_system.add_overlay("dev", "nn-cv")
    gmm_system.crp["nn-cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["nn-cv"].segment_path = cv_segments
    gmm_system.crp["nn-cv"].concurrent = 1
    gmm_system.crp["nn-cv"].corus_duration = DURATIONS["dev"] * cv_size

    gmm_system.add_overlay("train", "nn-devtrain")
    gmm_system.crp["nn-devtrain"].segment_path = devtrain_segments
    gmm_system.crp["nn-devtrain"].concurrent = 1
    gmm_system.crp["nn-devtrain"].corpus_config.file = devtrain_corpus_path
    gmm_system.crp["nn-devtrain"].corus_duration = DURATIONS["train"] * dev_train_size

    # ******************** extract features ********************

    train_features, cv_features, devtrain_features = dump_features_for_hybrid_training(
        gmm_system,
        feature_extraction_args,
        feature_extraction_class,
    )
    states = DumpStateTyingJob(gmm_system.crp["train"])
    allophone_labeling = AllophoneLabeling(
        silence_phoneme="[SILENCE]",
        allophone_file=gmm_system.allophone_files["train"],
        state_tying_file=states.out_state_tying,
    )

    forced_align_args = get_align_dev_args(crp=cv_corpus_path)
    gmm_system.run_forced_align_step(forced_align_args)

    nn_train_data = build_hdf_data_input(
        features=train_features["gt"],
        alignments=gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        alias_prefix=alias_prefix,
    )
    tk.register_output(f"{alias_prefix}/nn_train_data/features", nn_train_data.features)
    tk.register_output(f"{alias_prefix}/nn_train_data/alignments", nn_train_data.alignments)
    nn_devtrain_data = build_hdf_data_input(
        features=devtrain_features["gt"],
        alignments=gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        segment_list=devtrain_segments,
        alias_prefix=alias_prefix,
    )
    tk.register_output(f"{alias_prefix}/nn_devtrain_data/features", nn_devtrain_data.features)
    tk.register_output(f"{alias_prefix}/nn_devtrain_data/alignments", nn_devtrain_data.alignments)
    nn_cv_data = build_hdf_data_input(
        features=cv_features["gt"],
        alignments=gmm_system.alignments["nn-cv_forced-align"]["nn-cv"].alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        alias_prefix=alias_prefix,
    )
    tk.register_output(f"{alias_prefix}/nn_cv_data/features", nn_cv_data.features)
    tk.register_output(f"{alias_prefix}/nn_cv_data/alignments", nn_cv_data.alignments)

    nn_train_data_inputs = {
        "train.train": nn_train_data,
    }
    nn_devtrain_data_inputs = {
        "train.devtrain": nn_devtrain_data,
    }

    nn_cv_data_inputs = {
        "dev.cv": nn_cv_data,
    }

    nn_dev_data_inputs = {
        "dev": gmm_system.outputs["dev"]["final"].as_returnn_rasr_data_input(),
    }
    nn_test_data_inputs = {
        # "test": gmm_system.outputs["test"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
    }

    return (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    )
