from typing import Any, Callable, Dict, Tuple, Type, Union
from enum import Enum

from sisyphus import tk, delayed_ops

import i6_core.corpus as corpus_recipe
import i6_core.returnn as returnn
import i6_core.text as text

from i6_core.features import FeatureExtractionJob

from i6_experiments.common.datasets.librispeech import durations, num_segments
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.setups.rasr.util.nn import AllophoneLabeling, HdfDataInput, ReturnnRasrDataInput

from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.gmm.baseline_args import get_align_dev_args

from experimental.rasr.archiver import ArchiverJob

from i6_experiments.users.luescher.experiments.baselines.librispeech.default_tools import RASR_BINARY_PATH, SCTK_BINARY_PATH, RETURNN_EXE_PATH, RETURNN_ROOT_PATH, RETURNN_COMMON_PATH


class CvSplit(Enum):
    FROM_DEV = "dev"
    FROM_TRAIN = "train"


def get_corpora_for_hybrid_training(
    gmm_system: GmmSystem, cv_split: CvSplit = CvSplit.FROM_DEV
) -> Dict[str, Tuple[tk.Path, tk.Path, tk.Path]]:
    """
    split corpora for hybrid training
    :param gmm_system:
    :param cv_split:
    :return bliss corpora paths:
    """
    train_corpus_path = gmm_system.corpora["train-other-960"].corpus_file
    dev_clean_corpus_path = gmm_system.corpora["dev-clean"].corpus_file
    dev_other_corpus_path = gmm_system.corpora["dev-other"].corpus_file

    no_oov_dev_clean_corpus_path = corpus_recipe.FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=dev_clean_corpus_path,
        bliss_lexicon=gmm_system.crp["train-other-960"].lexicon_config.file,
        case_sensitive=False,
        all_unknown=False,
    ).out_corpus
    no_oov_dev_other_corpus_path = corpus_recipe.FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=dev_other_corpus_path,
        bliss_lexicon=gmm_system.crp["train-other-960"].lexicon_config.file,
        case_sensitive=False,
        all_unknown=False,
    ).out_corpus

    total_train_num_segments = num_segments["train-other-960"]
    total_dev_clean_num_segments = num_segments["dev-clean"]
    total_dev_other_num_segments = num_segments["dev-other"]

    all_dev_clean_segments_job = corpus_recipe.SegmentCorpusJob(no_oov_dev_clean_corpus_path, 1)
    all_dev_other_segments_job = corpus_recipe.SegmentCorpusJob(no_oov_dev_other_corpus_path, 1)

    all_train_segments = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1).out_single_segment_files[1]
    all_dev_clean_segments = all_dev_clean_segments_job.out_single_segment_files[1]
    all_dev_other_segments = all_dev_other_segments_job.out_single_segment_files[1]

    no_oov_dev_clean_num_segments = corpus_recipe.CountSegmentsInCorpusJob(no_oov_dev_clean_corpus_path).out_num_segments
    tk.register_output("dev_clean_no_oov_num_segments.txt", no_oov_dev_clean_num_segments)
    no_oov_dev_other_num_segments = corpus_recipe.CountSegmentsInCorpusJob(no_oov_dev_other_corpus_path).out_num_segments
    tk.register_output("dev_other_no_oov_num_segments.txt", no_oov_dev_other_num_segments)

    dev_train_size = 300 / total_train_num_segments
    cv_clean_size = 150 / total_dev_clean_num_segments
    cv_other_size = 150 / total_dev_other_num_segments

    if cv_split == CvSplit.FROM_DEV:
        split_train_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_train_segments,
            {"devtrain": dev_train_size, "unused": 1 - dev_train_size},
        )
        split_dev_clean_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_dev_clean_segments,
            {"cv": cv_clean_size, "unused": 1 - cv_clean_size},
        )
        split_dev_other_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_dev_other_segments,
            {"cv": cv_other_size, "unused": 1 - cv_other_size},
        )
        devtrain_segments = split_train_segments_job.out_segments["devtrain"]
        dev_clean_segments = split_dev_clean_segments_job.out_segments["cv"]
        dev_other_segments = split_dev_other_segments_job.out_segments["cv"]

        cv_segments = text.PipelineJob([dev_clean_segments, dev_other_segments], [], mini_task=True).out

        merged_dev_corpus_path = corpus_recipe.MergeCorporaJob(
            [dev_clean_corpus_path, dev_other_corpus_path],
            "dev-clean-other",
            corpus_recipe.MergeStrategy.FLAT,
        ).out_merged_corpus

        cv_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(merged_dev_corpus_path, cv_segments).out_corpus

        devtrain_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(train_corpus_path, devtrain_segments).out_corpus

    elif cv_split == CvSplit.FROM_TRAIN:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return {
        "corpora": (train_corpus_path, cv_corpus_path, devtrain_corpus_path),
        "segments": (all_train_segments, cv_segments, devtrain_segments),
        "durations": (
            gmm_system.crp["train-other-960"].corpus_duration,
            cv_clean_size * gmm_system.crp["dev-clean"].corpus_duration
            + cv_other_size * gmm_system.crp["dev-other"].corpus_duration,
            dev_train_size * gmm_system.crp["train-other-960"].corpus_duration,
        ),
    }


def dump_features_for_hybrid_training(
    gmm_system: GmmSystem, feature_extraction_args: Dict[str, Any], feature_extraction_class: Callable[[Any, ...], FeatureExtractionJob]
) -> Tuple[Dict[int, tk.Path], Dict[int, tk.Path], Dict[int, tk.Path]]:
    features = {}
    for name in ["nn-train", "nn-cv", "nn-devtrain"]:
        features[name] = feature_extraction_class(
            gmm_system.crp[name], **feature_extraction_args
        ).out_single_feature_caches

    return features["nn-train"], features["nn-cv"], features["nn-devtrain"]


def dump_features_into_hdf(
    features: Dict[int, tk.Path], allophone_labeling: AllophoneLabeling, filter_keep_list: tk.Path
) -> tk.Path:
    dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": list(features.values()),
                "data_type": "feat",
                "allophone_labeling": allophone_labeling,
            },
        },
        "seq_list_filter_file": filter_keep_list,
    }

    hdf_file = returnn.ReturnnDumpHDFJob(
        dataset,
        returnn_python_exe=RETURNN_EXE_PATH,
        returnn_root=RETURNN_ROOT_PATH,
    ).out_hdf

    return hdf_file


def dump_alignments_into_hdf(
    alignments: Dict[str, tk.Path], allophone_labeling: AllophoneLabeling, filter_keep_list: tk.Path
) -> tk.Path:
    dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": list(alignments.values()),
                "data_type": "align",
                "allophone_labeling": allophone_labeling,
            },
        },
        "seq_list_filter_file": filter_keep_list,
    }

    hdf_file = returnn.ReturnnDumpHDFJob(
        dataset,
        returnn_python_exe=RETURNN_EXE_PATH,
        returnn_root=RETURNN_ROOT_PATH,
    ).out_hdf

    return hdf_file


def get_corpus_data_inputs(
    gmm_system: GmmSystem,
    feature_extraction_args: Dict[str, Any],
    feature_extraction_class: Callable[[Any, ...], FeatureExtractionJob],
) -> Tuple[
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, ReturnnRasrDataInput],
    Dict[str, ReturnnRasrDataInput],
]:
    """
    get corpus files and split via segment files

    5 data dicts as output
    - train: full train
    - cv: subset from dev-clean and dev-other
    - devtrain: subset from train
    - dev
    - test

    :param gmm_system:
    :param feature_extraction_args:
    :param feature_extraction_class:
    :return:
    """

    corpora_segments_dict = get_corpora_for_hybrid_training(gmm_system)
    train_corpus_path, cv_corpus_path, devtrain_corpus_path = corpora_segments_dict["corpora"]
    train_segments, cv_segments, devtrain_segments = corpora_segments_dict["segments"]
    train_durations, cv_durations, devtrain_durations = corpora_segments_dict["durations"]

    gmm_system.add_overlay("train-other-960", "nn-train")
    gmm_system.add_overlay("dev-other", "nn-cv")
    gmm_system.add_overlay("train-other-960", "nn-devtrain")

    gmm_system.crp["nn-train"].segment_path = train_segments
    gmm_system.crp["nn-train"].concurrent = 1

    gmm_system.crp["nn-cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["nn-cv"].segment_path = cv_segments
    gmm_system.crp["nn-cv"].corpus_duration = cv_durations
    gmm_system.crp["nn-cv"].concurrent = 1

    gmm_system.crp["nn-devtrain"].corpus_config.file = devtrain_corpus_path
    gmm_system.crp["nn-devtrain"].segment_path = devtrain_segments
    gmm_system.crp["nn-devtrain"].corpus_duration = devtrain_durations
    gmm_system.crp["nn-devtrain"].concurrent = 1

    # ******************** extract features ********************

    train_features, cv_features, devtrain_features = dump_features_for_hybrid_training(
        gmm_system,
        feature_extraction_args,
        feature_extraction_class,
    )

    allophone_labeling = AllophoneLabeling(
        silence_phoneme="[SILENCE]",
        allophone_file=gmm_system.allophone_files["train-other-960"],
        state_tying_file=gmm_system.cart_trees["train-other-960"]["cart_mono"],
    )

    # ******************** dump features ********************

    train_feat_hdf = dump_features_into_hdf(train_features, allophone_labeling, train_segments)
    cv_feat_hdf = dump_features_into_hdf(cv_features, allophone_labeling, cv_segments)
    devtrain_feat_hdf = dump_features_into_hdf(devtrain_features, allophone_labeling, devtrain_segments)

    tk.register_output("train.feat.hdf", train_feat_hdf)
    tk.register_output("cv.feat.hdf", cv_feat_hdf)
    tk.register_output("devtrain.feat.hdf", devtrain_feat_hdf)

    # ******************** dump alignments ********************

    train_alignments = devtrain_alignments = (
        gmm_system.outputs["train-other-960"]["final"]
        .as_returnn_rasr_data_input()
        .alignments.alternatives["task_dependent"]
        .hidden_paths
    )

    forced_align_args = get_align_dev_args(name="nn-cv", target_corpus_keys=["nn-cv"])
    gmm_system.run_forced_align_step(forced_align_args)

    cv_alignments = gmm_system.alignments["nn-cv_forced-align"]["nn-cv"].alternatives["task_dependent"].hidden_paths

#    dev_clean_alignments = gmm_system.alignments["nn-cv_forced-align"]["nn-cv"].alternatives["task_dependent"].hidden_paths
#    dev_other_alignments = gmm_system.alignments["dev-other"]["nn-cv"].alternatives["task_dependent"].hidden_paths
#
#    cv_alignments: Dict[str, tk.Path] = {}
#    for k, v in dev_clean_alignments.items():
#        cv_alignments[f"dev-clean-{k}"] = v
#    for k, v in dev_other_alignments.items():
#        cv_alignments[f"dev-other-{k}"] = v

    train_align_hdf = dump_alignments_into_hdf(train_alignments, allophone_labeling, train_segments)
    cv_align_hdf = dump_alignments_into_hdf(cv_alignments, allophone_labeling, cv_segments)
    devtrain_align_hdf = dump_alignments_into_hdf(devtrain_alignments, allophone_labeling, devtrain_segments)

    tk.register_output("train.align.hdf", train_align_hdf)
    tk.register_output("cv.align.hdf", cv_align_hdf)
    tk.register_output("devtrain.align.hdf", devtrain_align_hdf)

    train_nn_data = HdfDataInput(
        [train_feat_hdf],
        [train_align_hdf],
        partition_epoch=40,
        seq_ordering="laplace:.1000",
    )
    cv_nn_data = HdfDataInput(
        [cv_feat_hdf],
        [cv_align_hdf],
        partition_epoch=1,
        seq_ordering="sorted",
    )
    devtrain_nn_data = HdfDataInput(
        [devtrain_feat_hdf],
        [devtrain_align_hdf],
        partition_epoch=1,
        seq_ordering="sorted",
    )

    nn_train_data_inputs = {
        "train-other-960.train": train_nn_data,
    }
    nn_devtrain_data_inputs = {
        "train-other-960.devtrain": devtrain_nn_data,
    }

    nn_cv_data_inputs = {
        "dev-clean-other.cv": cv_nn_data,
    }

    nn_dev_data_inputs = {
        # "dev-clean": gmm_system.outputs["dev-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        "dev-other": gmm_system.outputs["dev-other"]["final"].as_returnn_rasr_data_input(),
    }
    nn_test_data_inputs = {
        # "test-clean": gmm_system.outputs["test-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        # "test-other": gmm_system.outputs["test-other"][
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
