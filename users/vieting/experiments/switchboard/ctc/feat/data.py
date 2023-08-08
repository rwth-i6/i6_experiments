from typing import Dict

from sisyphus import Path
from i6_core import corpus as corpus_recipe
from i6_core import text
from i6_core.returnn.config import CodeWrapper
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_experiments.common.datasets.switchboard.corpus_eval import get_hub5e00
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.setups.rasr.util import OggZipHdfDataInput


def get_corpus_data_inputs_oggzip(
    gmm_system,
    partition_epoch,
    context_window=None,
    returnn_root=None,
    returnn_python_exe=None,
    pre_process=None
):
    """
    :param GmmSystem gmm_system:
    :param Dict[str, int] partition_epoch:
    :param Optional[Dict[str, int]] context_window:
    :param Optional[Path] returnn_root:
    :param Optional[Path] returnn_python_exe:
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

    blacklisted_segments = [
        "switchboard-1/sw02986A/sw2986A-ms98-a-0013",
        "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
        "switchboard-1/sw02691A/sw2691A-ms98-a-0017",
        "switchboard-1/sw04091A/sw4091A-ms98-a-0063",
        "switchboard-1/sw04103A/sw4103A-ms98-a-0022",
        "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
        "switchboard-1/sw04318A/sw4318A-ms98-a-0024",
    ]
    train_segments = corpus_recipe.FilterSegmentsByListJob(
        segment_files={1: train_segments},
        filter_list=blacklisted_segments,
    ).out_single_segment_files[1]
    cv_segments = corpus_recipe.FilterSegmentsByListJob(
        segment_files={1: cv_segments},
        filter_list=blacklisted_segments,
    ).out_single_segment_files[1]
    devtrain_segments = text.TailJob(
        train_segments, num_lines=300, zip_output=False
    ).out
    traincv_segments = corpus_recipe.FilterSegmentsByListJob(
        segment_files={1: all_segments},
        filter_list=blacklisted_segments,
    ).out_single_segment_files[1]

    # oggzip
    segments = corpus_recipe.SplitSegmentFileJob(all_segments, concurrent=20).out_segment_path
    ogg_zip_job = BlissToOggZipJob(
        train_corpus_path,
        raw_sample_rate=8000,
        feat_sample_rate=100,
        segments=segments,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    ogg_zip_job.rqmt = {"time": 8.0, "cpu": 2}
    meta_args = {"data_map": {"classes": ("hdf", "data"), "data": ("ogg", "data")}}
    if context_window is not None:
        meta_args["context_window"] = context_window
    audio = {
            "features": "raw",
            "peak_normalization": True
        }
    if pre_process is not None:
        audio["pre_process"] = pre_process
    ogg_zip_base_args = dict(
        oggzip_files=[ogg_zip_job.out_ogg_zip],
        alignments=[],
        audio=audio,
        meta_args=meta_args,
        acoustic_mixtures=gmm_system.outputs["switchboard"]["final"].acoustic_mixtures,
    )

    # nn data
    assert set(partition_epoch.keys()) == {"train", "dev"}
    nn_train_data = OggZipHdfDataInput(
        partition_epoch=partition_epoch["train"],
        ogg_args={"segment_file": train_segments, "targets": None},
        seq_ordering="laplace:.384",
        **ogg_zip_base_args,
    )
    nn_train_data_inputs = {
        "switchboard.train": nn_train_data,
    }

    nn_cv_data = OggZipHdfDataInput(
        partition_epoch=partition_epoch["dev"],
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": cv_segments, "targets": None},
        **ogg_zip_base_args,
    )
    nn_cv_data_inputs = {
        "switchboard.cv": nn_cv_data,
    }

    nn_devtrain_data = OggZipHdfDataInput(
        partition_epoch=partition_epoch["dev"],
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": devtrain_segments, "targets": None},
        **ogg_zip_base_args,
    )
    nn_devtrain_data_inputs = {
        "switchboard.devtrain": nn_devtrain_data,
    }

    hub5e00 = get_hub5e00()
    hub5e00_data = gmm_system.outputs["hub5e00"]["final"].as_returnn_rasr_data_input()
    hub5e00_data.stm = hub5e00.stm
    hub5e00_data.glm = hub5e00.glm
    nn_dev_data_inputs = {"hub5e00": hub5e00_data}
    nn_test_data_inputs = {}

    return (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
        train_corpus_path,
        traincv_segments,
    )
