from i6_core import corpus as corpus_recipe
from i6_core import text

from i6_experiments.common.setups.rasr.gmm_system import GmmSystem


def get_corpus_data_inputs(gmm_system):
    """

    :param GmmSystem gmm_system:
    :return:
    """


    train_corpus_path = gmm_system.corpora["train-clean-100"].corpus_file
    total_train_num_segments = 28539
    cv_size = 3000 / total_train_num_segments

    all_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - cv_size, "cv": cv_size}
    )
    train_segments = splitted_segments_job.out_segments["train"]
    cv_segments = splitted_segments_job.out_segments["cv"]
    devtrain_segments = text.TailJob(
        train_segments, num_lines=1000, zip_output=False
    ).out

    # ******************** NN Init ********************

    nn_train_data = gmm_system.outputs["train-clean-100"][
        "final"
    ].as_returnn_rasr_data_input(shuffle_data=True)
    nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
    # nn_train_data.crp.set_executables(rasr_root="/work/tools/asr/rasr/20211217_tf23_cuda101_mkl/")
    nn_train_data_inputs = {
        "train-clean-100.train": nn_train_data,
    }

    nn_cv_data = gmm_system.outputs["train-clean-100"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
    # nn_cv_data.crp.set_executables(rasr_root="/work/tools/asr/rasr/20211217_tf23_cuda101_mkl/")
    nn_cv_data_inputs = {
        "train-clean-100.cv": nn_cv_data,
    }

    nn_devtrain_data = gmm_system.outputs["train-clean-100"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    # nn_devtrain_data.crp.set_executables(rasr_root="/work/tools/asr/rasr/20211217_tf23_cuda101_mkl/")
    nn_devtrain_data_inputs = {
        "train-clean-100.devtrain": nn_devtrain_data,
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


    return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs, nn_dev_data_inputs, nn_test_data_inputs