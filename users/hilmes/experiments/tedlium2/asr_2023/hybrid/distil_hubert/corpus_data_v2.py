from typing import Optional, Dict, Any, Tuple, Callable
from sisyphus import tk

from i6_core import corpus as corpus_recipe
from i6_core.returnn import ReturnnDumpHDFJob
from i6_core.features import FeatureExtractionJob

from i6_experiments.common.datasets.tedlium2.constants import DURATIONS, NUM_SEGMENTS
from i6_experiments.users.hilmes.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.hilmes.common.datasets.tedlium2.corpus import get_ogg_zip_dict
from i6_experiments.users.hilmes.common.setups.rasr.util import (
    AllophoneLabeling,
    ReturnnRasrDataInput,
    ForcedAlignmentArgs,
)
from i6_experiments.common.datasets.tedlium2.lexicon import get_g2p_augmented_bliss_lexicon
from i6_experiments.users.hilmes.common.tedlium2.default_tools import RETURNN_EXE, RETURNN_RC_ROOT


def build_data_input(
    allophone_labeling: AllophoneLabeling,
    alignments: tk.Path,
    raw_features: tk.Path,
    segment_list: Optional[tk.Path] = None,
    alias_prefix: Optional[str] = None,
    partition_epoch: int = 1,
    seq_ordering: str = "sorted",
):
    """
    Dumps features and alignments from RASR into hdfs, to enable full RETURNN training
    :param allophone_labeling: Allophone labeling including silence_phoneme, allophones and state_tying
    :param alignments: Target alignments generated from the pre-trained GMM
    :param segment_list: segment list for the alignment dataset which will serve as seq_control dataset
    :param alias_prefix: Prefix for the dump jobs
    :param partition_epoch: Partition epoch for the alignment dataset, mainly relevant for training dataset
    :param seq_ordering: sequence ordering for the align dataset, usually sorted for dev/eval and laplace for train
    :return:
    """

    align_dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": alignments,
                "data_type": "align",
                "allophone_labeling": {
                    "silence_phone": allophone_labeling.silence_phone,
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            }
        },
        "seq_list_filter_file": segment_list,
    }
    align_job = ReturnnDumpHDFJob(data=align_dataset, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_RC_ROOT)
    if alias_prefix is not None:
        align_job.add_alias(alias_prefix + "/dump_alignments")
    align_hdf = align_job.out_hdf

    return {
            "class": "MetaDataset",
            "data_map": {"classes": ("hdf_align", "data"), "data_raw": ("ogg", "data")},
            "datasets": {
                "hdf_align": {
                    "class": "HDFDataset",
                    "files": [align_hdf],
                    "use_cache_manager": True,
                },
                "ogg": {
                    "class": "OggZipDataset",
                    "audio": {
                        "features": "raw",
                        "peak_normalization": True,
                        "sample_rate": 16000},
                    "partition_epoch": partition_epoch,
                    "path": [raw_features],
                    "seq_ordering": seq_ordering,
                    "use_cache_manager": True,
                    "segment_file": segment_list,
                    "targets": None,
                },
            },
            "seq_order_control_dataset": "ogg",
    }


def get_corpus_data_inputs(
    gmm_system: GmmSystem,
    alias_prefix: Optional[str] = None,
    remove_faulty_segments: bool = False,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, ReturnnRasrDataInput],
    Dict[str, ReturnnRasrDataInput],
]:
    """
    Builds the data inputs for the hybrid system, inlcuding 3 training hdf pairs with align and feature dataset for
    full returnn training
    :param gmm_system: Pre-trained GMM-system to derive the hybrid setup from
    :param alias_prefix: Prefix for naming of experiments
    :return:
    """

    train_corpus_path = gmm_system.corpora["train"].corpus_file
    cv_corpus_path = gmm_system.corpora["dev"].corpus_file

    cv_corpus_path = corpus_recipe.FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=cv_corpus_path, bliss_lexicon=get_g2p_augmented_bliss_lexicon(), all_unknown=False
    ).out_corpus

    total_train_num_segments = NUM_SEGMENTS["train"]

    all_train_segments = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1).out_single_segment_files[1]
    if remove_faulty_segments:
        all_train_segments = corpus_recipe.FilterSegmentsByListJob(
            segment_files={1: all_train_segments},
            filter_list=["TED-LIUM-realease2/AndrewMcAfee_2013/23", "TED-LIUM-realease2/iOTillettWright_2012X/43"],
        ).out_single_segment_files[1]
    cv_segments = corpus_recipe.SegmentCorpusJob(cv_corpus_path, 1).out_single_segment_files[1]

    dev_train_size = 500 / total_train_num_segments
    splitted_train_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_train_segments,
        {"devtrain": dev_train_size, "unused": 1 - dev_train_size},
    )
    devtrain_segments = splitted_train_segments_job.out_segments["devtrain"]

    # ******************** NN Init ********************

    gmm_system.add_overlay("train", "nn-train")
    gmm_system.crp["nn-train"].segment_path = all_train_segments
    gmm_system.crp["nn-train"].concurrent = 1
    gmm_system.crp["nn-train"].corpus_duration = DURATIONS["train"]

    gmm_system.add_overlay("dev", "nn-cv")
    gmm_system.crp["nn-cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["nn-cv"].segment_path = cv_segments
    gmm_system.crp["nn-cv"].concurrent = 1
    gmm_system.crp["nn-cv"].corpus_duration = DURATIONS["dev"]

    gmm_system.add_overlay("train", "nn-devtrain")
    gmm_system.crp["nn-devtrain"].segment_path = devtrain_segments
    gmm_system.crp["nn-devtrain"].concurrent = 1
    gmm_system.crp["nn-devtrain"].corpus_duration = DURATIONS["train"] * dev_train_size

    # ******************** extract features ********************

    extra_args = {
        c: {
            "rasr_cache": gmm_system.feature_bundles.get(c, {"mfcc": None})["mfcc"],
            "raw_sample_rate": 16000,
            "feat_sample_rate": 100,
        } for c in ["train", "dev", "test"]
    }
    ogg_zip_dict = get_ogg_zip_dict(
       extra_args=extra_args,
       returnn_root=RETURNN_RC_ROOT,
       returnn_python_exe=RETURNN_EXE)
    # RETURNN_ROOT = "/u/hilmes/dev/returnn/"
    # RETURNN_PYTHON_EXE = "/work/asr4/vieting/programs/conda/20230126/anaconda3/envs/py310_tf210/bin/python3"

    train_features_raw = ogg_zip_dict["train"]
    cv_features_raw = ogg_zip_dict["dev"]
    devtrain_features_raw = ogg_zip_dict["train"]

    allophone_labeling = AllophoneLabeling(
        silence_phone="[SILENCE]",
        allophone_file=gmm_system.allophone_files["train"],
        state_tying_file=gmm_system.jobs["train"]["state_tying"].out_state_tying,
    )

    forced_align_args = ForcedAlignmentArgs(
        name="nn-cv",
        target_corpus_keys=["nn-cv"],
        flow="uncached_mfcc+context+lda+vtln+cmllr",
        feature_scorer="train_vtln+sat",
        scorer_index=-1,
        bliss_lexicon={
            "filename": get_g2p_augmented_bliss_lexicon(),
            "normalize_pronunciation": False,
        },
        dump_alignment=True,
    )
    gmm_system.run_forced_align_step(forced_align_args)

    nn_train_data = build_data_input(
        alignments=gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        alias_prefix=alias_prefix + "/nn_train_data",
        partition_epoch=5,
        #acoustic_mixtures=gmm_system.outputs["train"]["final"].acoustic_mixtures,  # TODO: NN Mixtures
        seq_ordering="laplace:.1000",
        raw_features=train_features_raw,
    )
    #tk.register_output(f"{alias_prefix}/nn_train_data/out.ogg.zip", nn_train_data.oggzip_files[0])
    #tk.register_output(f"{alias_prefix}/nn_train_data/alignments", nn_train_data.alignments[0])
    nn_devtrain_data = build_data_input(
        alignments=gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        segment_list=devtrain_segments,
        alias_prefix=alias_prefix + "/nn_devtrain_data",
        partition_epoch=1,
        seq_ordering="sorted",
        raw_features=devtrain_features_raw
    )
    #tk.register_output(f"{alias_prefix}/nn_devtrain_data/out.ogg.zip", nn_devtrain_data.oggzip_files[0])
    #tk.register_output(f"{alias_prefix}/nn_devtrain_data/alignments", nn_devtrain_data.alignments[0])
    nn_cv_data = build_data_input(
        alignments=gmm_system.alignments["nn-cv_forced-align"]["nn-cv"].alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        alias_prefix=alias_prefix + "/nn_cv_data",
        partition_epoch=1,
        seq_ordering="sorted",
        segment_list=cv_segments,  # TODO REMOVE
        raw_features=cv_features_raw
    )
    #tk.register_output(f"{alias_prefix}/nn_cv_data/out.ogg.zip", nn_cv_data.oggzip_files[0])
    #tk.register_output(f"{alias_prefix}/nn_cv_data/alignments", nn_cv_data.alignments[0])

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
