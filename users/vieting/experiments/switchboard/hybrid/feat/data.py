import numpy as np

from i6_core import corpus as corpus_recipe
from i6_core import text
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_experiments.common.datasets.switchboard.corpus_eval import get_hub5e00
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.setups.rasr.util import OggZipHdfDataInput


def get_corpus_data_inputs_newcv(gmm_system):
    """

    :param GmmSystem gmm_system:
    :return:
    """
    train_corpus_path = gmm_system.corpora["switchboard"].corpus_file
    total_train_num_segments = 249536
    cv_size = 300 / total_train_num_segments

    all_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - cv_size, "cv": cv_size}
    )
    train_segments = splitted_segments_job.out_segments["train"]
    cv_segments = splitted_segments_job.out_segments["cv"]
    devtrain_segments = text.TailJob(
        train_segments, num_lines=300, zip_output=False
    ).out

    # ******************** NN Init ********************

    nn_train_data = gmm_system.outputs["switchboard"][
        "final"
    ].as_returnn_rasr_data_input(shuffle_data=True)
    nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
    nn_train_data_inputs = {
        "switchboard.train": nn_train_data,
    }

    nn_cv_data = gmm_system.outputs["switchboard"]["final"].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
    nn_cv_data_inputs = {
        "switchboard.cv": nn_cv_data,
    }

    nn_devtrain_data = gmm_system.outputs["switchboard"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    nn_devtrain_data_inputs = {
        "switchboard.devtrain": nn_devtrain_data,
    }

    hub5e00 = get_hub5e00()
    hub5e00_data = gmm_system.outputs["hub5e00"]["final"].as_returnn_rasr_data_input()
    hub5e00_data.stm = hub5e00.stm
    hub5e00_data.glm = hub5e00.glm
    nn_dev_data_inputs = {"hub5e00": hub5e00_data}
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


def get_corpus_data_inputs_oggzip(gmm_system, returnn_root=None, returnn_python_exe=None):
    """

    :param GmmSystem gmm_system:
    :return:
    """
    # create train and cv sets
    train_corpus_path = gmm_system.corpora["switchboard"].corpus_file
    total_train_num_segments = 249536
    cv_size = 300 / total_train_num_segments

    all_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_segments, {"train": 1 - cv_size, "cv": cv_size}
    )
    train_segments = splitted_segments_job.out_segments["train"]
    cv_segments = splitted_segments_job.out_segments["cv"]
    devtrain_segments = text.TailJob(
        train_segments, num_lines=300, zip_output=False
    ).out

    # alignment hdf and oggzip
    state_tying_job = DumpStateTyingJob(gmm_system.outputs["switchboard"]["final"].crp)
    allophone_file = gmm_system.outputs["switchboard"]["final"].crp.acoustic_model_post_config.allophones.add_from_file
    train_align_job = RasrAlignmentDumpHDFJob(
        alignment_caches=list(gmm_system.outputs["switchboard"]["final"].alignments.hidden_paths.values()),
        state_tying_file=state_tying_job.out_state_tying,
        allophone_file=allophone_file,
        data_type=np.int16,
        returnn_root=returnn_root,
    )
    ogg_zip_job = BlissToOggZipJob(
        train_corpus_path,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    ogg_zip_base_args = dict(
        oggzip_files=[ogg_zip_job.out_ogg_zip],
        alignments=train_align_job.out_hdf_files,
        context_window={"classes": 1, "data": 400},
        audio={"features": "raw", "peak_normalization": True},
        meta_args={"data_map": {"classes": ("hdf", "data"), "data": ("ogg", "data")}},
        acoustic_mixtures=gmm_system.outputs["switchboard"]["final"].acoustic_mixtures,
    )

    # nn data
    nn_train_data = OggZipHdfDataInput(
        partition_epoch=6,  # TODO: get from args
        ogg_args={"segment_file": train_segments},
        **ogg_zip_base_args,
    )
    nn_train_data_inputs = {
        "switchboard.train": nn_train_data,
    }

    nn_cv_data = OggZipHdfDataInput(
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": cv_segments},
        **ogg_zip_base_args,
    )
    nn_cv_data_inputs = {
        "switchboard.cv": nn_cv_data,
    }

    nn_devtrain_data = OggZipHdfDataInput(
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": devtrain_segments},
        **ogg_zip_base_args,
    )
    nn_devtrain_data_inputs = {
        "switchboard.devtrain": nn_devtrain_data,
    }

    # hub5e00 = get_hub5e00()
    # hub5e00_data = gmm_system.outputs["hub5e00"]["final"].as_returnn_rasr_data_input()
    # hub5e00_data.stm = hub5e00.stm
    # hub5e00_data.glm = hub5e00.glm
    nn_dev_data_inputs = None  # {"hub5e00": hub5e00_data}  # TODO: add hub5e00
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
