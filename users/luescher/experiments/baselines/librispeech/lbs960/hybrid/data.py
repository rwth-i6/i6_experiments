import copy
from enum import Enum
from typing import Any, Callable, Dict, Tuple, Type, Union
from dataclasses import asdict

from sisyphus import tk, delayed_ops

import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.text as text

from i6_core.features import FeatureExtractionJob

from i6_experiments.common.datasets.librispeech import durations, num_segments
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.setups.rasr.util.nn import AllophoneLabeling, HdfDataInput, ReturnnRasrDataInput

from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.gmm.baseline_args import get_align_dev_args

from experimental.rasr.archiver import ArchiverJob

from i6_experiments.users.luescher.experiments.baselines.librispeech.default_tools import (
    RASR_BINARY_PATH,
    SCTK_BINARY_PATH,
    RETURNN_EXE_PATH,
    RETURNN_ROOT_PATH,
    RETURNN_COMMON_PATH,
)


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

    no_oov_dev_clean_num_segments = corpus_recipe.CountSegmentsInCorpusJob(
        no_oov_dev_clean_corpus_path
    ).out_num_segments
    tk.register_output("dev_clean_no_oov_num_segments.txt", no_oov_dev_clean_num_segments)
    no_oov_dev_other_num_segments = corpus_recipe.CountSegmentsInCorpusJob(
        no_oov_dev_other_corpus_path
    ).out_num_segments
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

        filter_corpus_args = {
            "compressed": True,
            "invert_match": True,
            "delete_empty_recordings": True,
        }

        cv_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(
            merged_dev_corpus_path, cv_segments, **filter_corpus_args
        ).out_corpus

        devtrain_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(
            train_corpus_path, devtrain_segments, **filter_corpus_args
        ).out_corpus

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


def dump_features_to_hdf(
    crp: rasr.CommonRasrParameters,
    flow_callable: Callable[[Any, ...], rasr.FlowNetwork],
    feature_extraction_options: Dict,
) -> tk.Path:
    feat_ext_opt = copy.deepcopy(feature_extraction_options)

    if "samples_options" not in feat_ext_opt:
        feat_ext_opt["samples_options"] = {}
    feat_ext_opt["samples_options"]["audio_format"] = crp.audio_format
    flow_name = feat_ext_opt.pop("name")

    feature_flow = features.feature_extraction_cache_flow(flow_callable(**feat_ext_opt), {"features": flow_name}, None)
    flow_path = rasr.WriteFlowNetworkJob(feature_flow).out_flow_file

    config, post_config = rasr.build_config_from_mapping(
        crp=crp,
        mapping={"corpus": "extraction.corpus"},
        parallelize=True,
    )
    config.extraction.feature_extraction.file = flow_path
    config.extraction.feature_extraction["*"].allow_overwrite = True
    feature_flow.apply_config("extraction.feature-extraction", config, post_config)
    rasr_config_path = rasr.WriteRasrConfigJob(config, post_config).out_config

    feature_hdf_path = returnn.ReturnnDumpHDFJob(
        data={
            "class": "ExternSprintDataset",
            # "sprintTrainerExecPath": rasr.RasrCommand.select_exe(crp.feature_extraction_exe, "feature-extraction"),
            "sprintTrainerExecPath": rasr.RasrCommand.select_exe(crp.nn_trainer_exe, "nn-trainer"),
            "sprintConfigStr": delayed_ops.DelayedFormat(
                "--config={} --*.LOGFILE=rasr.log --*.TASK=1", rasr_config_path
            ),
            "partitionEpoch": 1,
        },
        returnn_python_exe=RETURNN_EXE_PATH,
        returnn_root=RETURNN_ROOT_PATH,
    ).out_hdf

    return feature_hdf_path


def dump_features_for_hybrid_training(
    gmm_system: GmmSystem,
    feature_extraction_args: Dict[str, Any],
    feature_extraction_class: Callable[[Any, ...], FeatureExtractionJob],
) -> Tuple[tk.Path, tk.Path, tk.Path]:
    features = {}
    for name in ["nn-train", "nn-cv", "nn-devtrain"]:
        features[name] = list(
            feature_extraction_class(gmm_system.crp[name], **feature_extraction_args).out_feature_bundle.values()
        )[0]

    return features["nn-train"], features["nn-cv"], features["nn-devtrain"]


def dump_feature_cache_to_hdf(
    feature_bundle_path: tk.Path, allophone_labeling: AllophoneLabeling, filter_keep_list: tk.Path
) -> tk.Path:
    dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": delayed_ops.DelayedFormat("{}", feature_bundle_path),
                "data_type": "feat",
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
    alignment_bundle_path: tk.Path, allophone_labeling: AllophoneLabeling, filter_keep_list: tk.Path
) -> tk.Path:
    dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": delayed_ops.DelayedFormat("{}", alignment_bundle_path),
                "data_type": "align",
                "allophone_labeling": asdict(allophone_labeling),
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
    feature_extraction_flow: Callable[[Any, ...], rasr.FlowNetwork],
    feature_extraction_options: Dict[str, Any],
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
    :param feature_extraction_flow:
    :param feature_extraction_options:
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
        silence_phone="[SILENCE]",
        allophone_file=delayed_ops.DelayedFormat("{}", gmm_system.allophone_files["train-other-960"]),
        state_tying_file=delayed_ops.DelayedFormat(
            "{}", gmm_system.jobs["train-other-960"]["state_tying"].out_state_tying
        ),
    )

    # ******************** dump features ********************

    train_feat_hdf = dump_feature_cache_to_hdf(train_features, allophone_labeling, train_segments)
    cv_feat_hdf = dump_feature_cache_to_hdf(cv_features, allophone_labeling, cv_segments)
    devtrain_feat_hdf = dump_feature_cache_to_hdf(devtrain_features, allophone_labeling, devtrain_segments)

    tk.register_output("train.feat.hdf", train_feat_hdf)
    tk.register_output("cv.feat.hdf", cv_feat_hdf)
    tk.register_output("devtrain.feat.hdf", devtrain_feat_hdf)

    train_rasr_feat_hdf = dump_features_to_hdf(
        crp=gmm_system.crp["nn-train"],
        flow_callable=feature_extraction_flow,
        feature_extraction_options=feature_extraction_options,
    )
    cv_rasr_feat_hdf = dump_features_to_hdf(
        crp=gmm_system.crp["nn-cv"],
        flow_callable=feature_extraction_flow,
        feature_extraction_options=feature_extraction_options,
    )
    devtrain_rasr_feat_hdf = dump_features_to_hdf(
        crp=gmm_system.crp["nn-devtrain"],
        flow_callable=feature_extraction_flow,
        feature_extraction_options=feature_extraction_options,
    )

    tk.register_output("train.rasr_feat.hdf", train_rasr_feat_hdf)
    tk.register_output("cv.rasr_feat.hdf", cv_rasr_feat_hdf)
    tk.register_output("devtrain.rasr_feat.hdf", devtrain_rasr_feat_hdf)

    # ******************** dump alignments ********************

    train_alignments = devtrain_alignments = (
        gmm_system.outputs["train-other-960"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"]
    )

    forced_align_args = get_align_dev_args(name="nn-cv", target_corpus_keys=["nn-cv"])
    gmm_system.run_forced_align_step(forced_align_args)

    cv_alignments = gmm_system.alignments["nn-cv_forced-align"]["nn-cv"].alternatives["bundle"]

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
