from i6_core import corpus as corpus_recipe
from i6_core import text

from i6_experiments.common.datasets.tedlium2.constants import DURATIONS, NUM_SEGMENTS
from i6_experiments.users.luescher.setups.rasr.gmm_system import GmmSystem
#from i6_experiments.common.setups.rasr.util.nn import SingleHdfDataInput
from i6_experiments.users.luescher.setups.rasr.util.nn import AllophoneLabeling

from i6_experiments.users.luescher.experiments.baselines.librispeech.lbs960.gmm.baseline_args import get_align_dev_args


def get_corpus_data_inputs(gmm_system: GmmSystem):
    train_corpus_path = gmm_system.corpora["train"].corpus_file
    cv_corpus_path = gmm_system.corpora["dev"].corpus_file

    total_train_num_segments = NUM_SEGMENTS["train"]
    total_cv_num_segments = NUM_SEGMENTS["dev"]

    all_train_segments = corpus_recipe.SegmentCorpusJob(
        train_corpus_path, 1
    ).out_single_segment_files[1]

    all_cv_segments = corpus_recipe.SegmentCorpusJob(
        cv_corpus_path, 1
    ).out_single_segment_files[1]

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

    cv_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(
        cv_corpus_path, cv_segments
    ).out_corpus

    # TODO: remove this?
    train_allophone_labeling = AllophoneLabeling(
        silence_phoneme="[SILENCE]",
        allophone_file=gmm_system.allophone_files["train-other-960"],
        state_tying_file=gmm_system.cart_trees["train-other-960"]["cart_mono"],
    )

    # ******************** NN Init ********************

    nn_train_data = nn_devtrain_data = gmm_system.outputs["train"][
        "final"
    ].as_returnn_rasr_data_input()
    nn_train_data.update_crp_with(concurrent=1)
    nn_train_data_inputs = {
        "train": nn_train_data,
    }
    nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
    nn_devtrain_data_inputs = {
        "devtrain": nn_devtrain_data,
    }

    gmm_system.add_overlay("dev", "cv")
    gmm_system.crp["cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["cv"].segment_path = cv_segments
    gmm_system.crp["cv"].concurrent = 1
    gmm_system.feature_bundles["cv"] = gmm_system.feature_bundles["dev"]
    gmm_system.add_overlay("train", "devtrain")
    gmm_system.crp["devtrain"].segment_path = devtrain_segments

    forced_align_args = get_align_dev_args()
    gmm_system.run_forced_align_step(forced_align_args)

    forced_align_args = get_align_dev_args(name="cv", target_corpus_keys=["cv"])
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
    nn_cv_data_inputs = {
        "cv": nn_cv_data,
    }

    nn_dev_data_inputs = {
        "dev": gmm_system.outputs["dev"][
            "final"
        ].as_returnn_rasr_data_input(),
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
