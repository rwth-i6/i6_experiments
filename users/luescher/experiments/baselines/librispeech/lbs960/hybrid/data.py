from typing import Dict, Tuple

from i6_core import corpus as corpus_recipe
from i6_core import text

from i6_experiments.common.datasets.librispeech import durations, num_segments
from i6_experiments.users.luescher.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.luescher.setups.rasr.util.nn.data import HdfDataInput, RasrDataInput

# from i6_experiments.common.setups.rasr.util.nn import SingleHdfDataInput

from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.gmm.baseline_args import get_align_dev_args

from experimental.rasr.archiver import ArchiverJob

from .default_tools import RETURNN_EXE, RETURNN_RC_ROOT


def get_corpus_data_inputs(
    gmm_system: GmmSystem,
) -> Tuple[
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, RasrDataInput],
    Dict[str, RasrDataInput],
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
    :return:
    """
    train_corpus_path = gmm_system.corpora["train-other-960"].corpus_file
    dev_clean_corpus_path = gmm_system.corpora["dev-clean"].corpus_file
    dev_other_corpus_path = gmm_system.corpora["dev-other"].corpus_file

    total_train_num_segments = num_segments["train-other-960"]
    total_dev_clean_num_segments = num_segments["dev-clean"]
    total_dev_other_num_segments = num_segments["dev-other"]

    all_train_segments = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1).out_single_segment_files[1]

    all_dev_clean_segments = corpus_recipe.SegmentCorpusJob(dev_clean_corpus_path, 1).out_single_segment_files[1]

    all_dev_other_segments = corpus_recipe.SegmentCorpusJob(dev_other_corpus_path, 1).out_single_segment_files[1]

    dev_train_size = 500 / total_train_num_segments
    cv_clean_size = 150 / total_dev_clean_num_segments
    cv_other_size = 150 / total_dev_other_num_segments

    splitted_train_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_train_segments,
        {"devtrain": dev_train_size, "unused": 1 - dev_train_size},
    )
    splitted_dev_clean_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_dev_clean_segments,
        {"cv": cv_clean_size, "unused": 1 - cv_clean_size},
    )
    splitted_dev_other_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_dev_other_segments,
        {"cv": cv_other_size, "unused": 1 - cv_other_size},
    )

    devtrain_segments = splitted_train_segments_job.out_segments["devtrain"]
    dev_clean_segments = splitted_dev_clean_segments_job.out_segments["cv"]
    dev_other_segments = splitted_dev_other_segments_job.out_segments["cv"]

    cv_segments = text.PipelineJob([dev_clean_segments, dev_other_segments], [], mini_task=True).out

    merged_dev_corpus_path = corpus_recipe.MergeCorporaJob(
        [dev_clean_corpus_path, dev_other_corpus_path],
        "dev-clean-other",
        corpus_recipe.MergeStrategy.FLAT,
    ).out_merged_corpus

    cv_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(merged_dev_corpus_path, cv_segments).out_corpus

    # ******************** NN Init ********************

    nn_train_data = nn_devtrain_data = gmm_system.outputs["train-other-960"]["final"].as_returnn_rasr_data_input()
    nn_train_data.update_crp_with(concurrent=1)
    nn_train_data_inputs = {
        "train-other-960.train": nn_train_data,
    }
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    nn_devtrain_data_inputs = {
        "train-other-960.devtrain": nn_devtrain_data,
    }

    gmm_system.add_overlay("dev-other", "cv")
    gmm_system.crp["cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["cv"].segment_path = cv_segments
    gmm_system.crp["cv"].concurrent = 1
    gmm_system.feature_bundles["cv"] = text.PipelineJob(
        [gmm_system.feature_bundles["dev-clean"], gmm_system.feature_bundles["dev-other"]],
        [],
        mini_task=True,
    ).out
    gmm_system.add_overlay("train-other-960", "devtrain")
    gmm_system.crp["devtrain"].segment_path = devtrain_segments

    forced_align_args = get_align_dev_args()
    gmm_system.run_forced_align_step(forced_align_args)

    forced_align_args = get_align_dev_args(name="cv", target_corpus_keys=["cv"])
    gmm_system.run_forced_align_step(forced_align_args)

    # merge_alignments_job = ArchiverJob()  # TODO

    nn_cv_data = gmm_system.outputs["dev-other"]["final"].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(
        corpus_file=cv_corpus_path,
        corpus_duration=(durations["dev-clean"] + durations["dev-other"]) * (cv_clean_size + cv_other_size),
        segment_path=cv_segments,
        concurrent=1,
    )
    nn_cv_data.alignments = gmm_system.alignments["cv_forced"]["cv"]
    nn_cv_data_inputs = {
        "dev-clean-other.cv": nn_cv_data,
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
