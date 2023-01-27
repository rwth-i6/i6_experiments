from i6_core import corpus as corpus_recipe
from i6_core import text

from i6_experiments.common.datasets.librispeech import durations, num_segments
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
#from i6_experiments.common.setups.rasr.util.nn import SingleHdfDataInput

from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.gmm.baseline_args import get_align_dev_args

from experimental.rasr.archiver import ArchiverJob

from .default_tools import RETURNN_EXE, RETURNN_RC_ROOT


def get_corpus_data_inputs(gmm_system: GmmSystem):
    train_corpus_path = gmm_system.corpora["train-other-960"].corpus_file
    dev_clean_corpus_path = gmm_system.corpora["clean"].corpus_file
    dev_other_corpus_path = gmm_system.corpora["other"].corpus_file

    total_train_num_segments = num_segments["train-other-960"]
    total_dev_clean_num_segments = num_segments["dev-clean"]
    total_dev_other_num_segments = num_segments["dev-other"]

    all_train_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    all_dev_clean_segments = corpus_recipe.SegmentCorpusJob(
        dev_clean_corpus_path, 1
    ).out_single_segment_files[1]

    all_dev_other_segments = corpus_recipe.SegmentCorpusJob(
        dev_other_corpus_path, 1
    ).out_single_segment_files[1]

    dev_train_size = 1000 / total_train_num_segments
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

    cv_segments = text.PipelineJob(
        [dev_clean_segments, dev_other_segments], [], mini_task=True
    ).out

    merged_dev_corpus_path = corpus_recipe.MergeCorporaJob(
        [dev_clean_corpus_path, dev_other_corpus_path],
        "dev-clean-other",
        corpus_recipe.MergeStrategy.FLAT,
    ).out_merged_corpus

    # ******************** NN Init ********************

    nn_train_data = nn_devtrain_data = gmm_system.outputs["train-other-960"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_train_data.update_crp_with(concurrent=1)
    nn_train_data_inputs = {
        "train-other-960.train": nn_train_data,
    }
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    nn_devtrain_data_inputs = {
        "train-other-960.devtrain": nn_devtrain_data,
    }

    forced_align_args = get_align_dev_args()
    gmm_system.run_forced_align_step(forced_align_args)

    merge_alignments_job = ArchiverJob()  # TODO

    nn_cv_data = gmm_system.outputs["dev-other"]["final"].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(
        corpus_file=merged_dev_corpus_path,
        corpus_duration=durations["dev-clean"] + durations["dev-other"],
        segment_path=cv_segments,
        concurrent=1,
    )
    nn_cv_data.alignments = None  # TODO
    nn_cv_data_inputs = {
        "dev-clean-other.cv": nn_cv_data,
    }

    nn_dev_data_inputs = {
        # "dev-clean": gmm_system.outputs["dev-clean"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
        "dev-other": gmm_system.outputs["dev-other"][
            "final"
        ].as_returnn_rasr_data_input(),
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
