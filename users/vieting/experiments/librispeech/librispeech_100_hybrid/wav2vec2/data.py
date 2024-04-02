import numpy as np
from sisyphus import tk

from i6_core import corpus as corpus_recipe
from i6_core import text
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.returnn import RasrAlignmentDumpHDFJob, BlissToOggZipJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.rasr.util import OggZipHdfDataInput
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict


def get_ls100_oggzip_hdf_data(gmm_system: GmmSystem):
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn", commit="45fad83c785a45fa4abfeebfed2e731dd96f960c").out_repository
    returnn_root.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

    state_tying = DumpStateTyingJob(gmm_system.outputs["train-clean-100"]["final"].crp)
    train_align_job = RasrAlignmentDumpHDFJob(
        alignment_caches=list(gmm_system.outputs["train-clean-100"]["final"].alignments.hidden_paths.values()),
        state_tying_file=state_tying.out_state_tying,
        allophone_file=gmm_system.outputs["train-clean-100"]["final"].crp.acoustic_model_post_config.allophones.add_from_file,
        data_type=np.int16,
        returnn_root=returnn_root,
        sparse=True,
    )

    ogg_zip_dict = get_ogg_zip_dict(returnn_python_exe=returnn_exe, returnn_root=returnn_root)
    ogg_zip_base_args = dict(
        alignments=train_align_job.out_hdf_files,
        audio={"features": "raw", "peak_normalization": True, "preemphasis": None},
        meta_args={
            "data_map": {"classes": ("hdf", "data"), "data": ("ogg", "data")},
            "context_window": {"classes": 1, "data": 400},
        },
        ogg_args={"targets": None},
        acoustic_mixtures=gmm_system.outputs["train-clean-100"]["final"].acoustic_mixtures,
    )
    nn_data_inputs = {}
    nn_data_inputs["train"] = OggZipHdfDataInput(
        oggzip_files=[ogg_zip_dict["train-clean-100"]],
        partition_epoch=3,
        **ogg_zip_base_args,
    )
    nn_data_inputs["cv"] = OggZipHdfDataInput(
        oggzip_files=[ogg_zip_dict["dev-clean"]],
        seq_ordering="sorted_reverse",
        **ogg_zip_base_args,
    )
    nn_data_inputs["devtrain"] = OggZipHdfDataInput(
        oggzip_files=[ogg_zip_dict["dev-clean"]],
        seq_ordering="sorted_reverse",
        **ogg_zip_base_args,
    )
    return nn_data_inputs


def get_ls100_oggzip_hdf_data_split_train_cv(gmm_system: GmmSystem, sync_ogg: bool = False):
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn", commit="45fad83c785a45fa4abfeebfed2e731dd96f960c").out_repository
    returnn_root.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

    train_corpus_path = gmm_system.corpora["train-clean-100"].corpus_file
    total_train_num_segments = 28539
    cv_size = 1000 / total_train_num_segments

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

    state_tying = DumpStateTyingJob(gmm_system.outputs["train-clean-100"]["final"].crp)
    train_align_job = RasrAlignmentDumpHDFJob(
        alignment_caches=list(gmm_system.outputs["train-clean-100"]["final"].alignments.hidden_paths.values()),
        state_tying_file=state_tying.out_state_tying,
        allophone_file=gmm_system.outputs["train-clean-100"]["final"].crp.acoustic_model_post_config.allophones.add_from_file,
        data_type=np.int16,
        returnn_root=returnn_root,
        sparse=True,
    )
    ogg_zip_file = get_ogg_zip_dict(returnn_python_exe=returnn_exe, returnn_root=returnn_root)["train-clean-100"]
    if sync_ogg:
        ogg_zip_file = BlissToOggZipJob(
            bliss_corpus=ogg_zip_file.creator.bliss_corpus,
            rasr_cache=gmm_system.outputs["train-clean-100"]["final"].features[
                "mfcc+context+lda+vtln"].hidden_paths[1].creator.out_feature_bundle["vtln"],
            raw_sample_rate=16000,
            feat_sample_rate=100,
            no_conversion=False,
            returnn_root=returnn_root,
        ).out_ogg_zip
    ogg_zip_base_args = dict(
        oggzip_files=[ogg_zip_file],
        alignments=train_align_job.out_hdf_files,
        audio={"features": "raw", "peak_normalization": True, "preemphasis": None},
        meta_args={
            "data_map": {"classes": ("hdf", "data"), "data": ("ogg", "data")},
            "context_window": {"classes": 1, "data": 400},
        },
        acoustic_mixtures=gmm_system.outputs["train-clean-100"]["final"].acoustic_mixtures,
    )
    nn_data_inputs = {}
    nn_data_inputs["train"] = OggZipHdfDataInput(
        partition_epoch=3,
        ogg_args={"segment_file": train_segments, "targets": None},
        **ogg_zip_base_args,
    )
    nn_data_inputs["cv"] = OggZipHdfDataInput(
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": cv_segments, "targets": None},
        **ogg_zip_base_args,
    )
    nn_data_inputs["devtrain"] = OggZipHdfDataInput(
        seq_ordering="sorted_reverse",
        ogg_args={"segment_file": devtrain_segments, "targets": None},
        **ogg_zip_base_args,
    )
    return nn_data_inputs