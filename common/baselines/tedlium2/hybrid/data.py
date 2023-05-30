from typing import Optional
from sisyphus import tk

from i6_core import corpus as corpus_recipe
from i6_core.returnn import ReturnnDumpHDFJob

from i6_experiments.common.datasets.tedlium2.constants import DURATIONS, NUM_SEGMENTS
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.setups.rasr.util import HdfDataInput, AllophoneLabeling
from i6_experiments.common.baselines.tedlium2.gmm.baseline_args import get_align_dev_args


def build_hdf_data_input(
    corpus_path: tk.Path,
    allophone_labeling: AllophoneLabeling,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignment_path: tk.Path,
    segment_list: Optional[tk.Path] = None,
):

    feat_dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": corpus_path.get(),
                "data_type": "feat",
                "allophone_labeling": {
                    "silence_phone": "[silence]",
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            },
        },
        "seq_list_filter_file": segment_list,
    }

    feat_hdf = ReturnnDumpHDFJob(
        feat_dataset,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_hdf
    align_dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": alignment_path,
                "data_type": "align",
                "allophone_labeling": {
                    "silence_phone": "[silence]",
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            },
        },
        "seq_list_filter_file": segment_list,
    }
    align_hdf = ReturnnDumpHDFJob(align_dataset, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root)

    return HdfDataInput(features=feat_hdf, alignments=align_hdf)


def get_corpus_data_inputs(gmm_system: GmmSystem, returnn_root: tk.Path, returnn_python_exe: tk.Path):
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

    # TODO: remove this?
    train_allophone_labeling = AllophoneLabeling(
        silence_phoneme="[SILENCE]",
        allophone_file=gmm_system.allophone_files["train"],
        state_tying_file=gmm_system.cart_trees["train"]["cart_mono"],
    )

    # ******************** NN Init ********************

    nn_train_data = nn_devtrain_data = gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input()
    nn_train_data.update_crp_with(concurrent=1)

    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)

    gmm_system.add_overlay("dev", "cv")
    gmm_system.crp["cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["cv"].segment_path = cv_segments
    gmm_system.crp["cv"].concurrent = 1
    gmm_system.feature_bundles["cv"] = gmm_system.feature_bundles["dev"]
    gmm_system.add_overlay("train", "devtrain")
    gmm_system.crp["devtrain"].segment_path = devtrain_segments

    forced_align_args = get_align_dev_args(crp=cv_corpus_path)
    gmm_system.run_forced_align_step(forced_align_args)

    # merge_alignments_job = ArchiverJob()  # TODO

    nn_cv_data = gmm_system.outputs["dev"]["final"].as_returnn_rasr_data_input()
    nn_cv_data.update_crp_with(
        corpus_file=cv_corpus_path,
        corpus_duration=DURATIONS["dev"] * cv_size,
        segment_path=cv_segments,
        concurrent=1,
    )
    nn_cv_data.alignments = gmm_system.alignments["cv_forced"]["cv"]

    nn_dev_data_inputs = {
        "dev": gmm_system.outputs["dev"]["final"].as_returnn_rasr_data_input(),
    }
    nn_test_data_inputs = {
        # "test": gmm_system.outputs["test"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
    }

    nn_train_data_inputs = build_hdf_data_input(
        corpus_path=train_corpus_path,
        alignment_path=gmm_system.alignments["train"]["vtln+sat"],
        allophone_labeling=train_allophone_labeling,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    nn_devtrain_data_inputs = build_hdf_data_input(
        corpus_path=train_corpus_path,
        alignment_path=gmm_system.alignments["train"]["vtln+sat"],
        allophone_labeling=train_allophone_labeling,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        segment_list=devtrain_segments,
    )
    nn_cv_data_inputs = build_hdf_data_input(
        corpus_path=cv_corpus_path,
        alignment_path=gmm_system.alignments["cv_forced"]["cv"],
        allophone_labeling=train_allophone_labeling,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    return (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    )
