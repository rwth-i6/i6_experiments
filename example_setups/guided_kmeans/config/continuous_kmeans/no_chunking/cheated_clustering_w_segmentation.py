from sisyphus import tk

import sys

from i6_experiments.example_setups.guided_kmeans.setup.clustering_config import (
    clustering,
    ClusteringCallbackConfig,
    LateInitConfig,
    StreamingStandardInitializerConfig,
    PickleCheatingCentroidInitializerConfig,
    PreloadCentroidsInitializerConfig,
)

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config, create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, SegmentFile, All
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.corpus_setup import phoneme_corpus, setup_corpus

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as _BASE_INPUT_DATA,
    FEATURES_LS100H_SEGMENTED,
    FEATURES_LS_CV as _CV_FEATURES,
    SEGMENTS_LS_CV as _CV_SEGMENTS,
    RASR_PATH_LIBRASR as rasr_path,
)
from i6_experiments.example_setups.guided_kmeans import tools

# ls-100 points to the segmented HDF in this config (cheating segmentation experiment)
input_data = {
    **_BASE_INPUT_DATA,
    "ls-100": {
        **_BASE_INPUT_DATA["ls-100"],
        "features": FEATURES_LS100H_SEGMENTED,
    },
}


def run():
    corpus_file = phoneme_corpus(setup_corpus("train-clean-100"))

    use_eow_phonemes = False
    # num_epochs = 20
    num_epochs = 10
    use_pruning = False

    input_data_key = "ls-100"
    initialization = "cheating"

    parameters = [
        (None, 1.0, 0.0, 0.0)
    ]
    

    recog_results = []
    recog_results_cv = []

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=_CV_FEATURES,
        sampling_method=SegmentFile(_CV_SEGMENTS),
        precomputed=True,
    )

    recognition_config_decode = create_recog_rasr_config(
        lm_scale=10000.0,
        emission_scale=1.0,
        transition_scale=None,
        loop_probability=0.3,
        silence_loop_probability=0.3,
        use_tree_search=False,
        max_beam_size=20000 if use_pruning else None,
        score_threshold=10000.0 if use_pruning else None,
        lm_order=3,
        use_eow_phonemes=use_eow_phonemes,
    )


    for subsampling, lm_scale, loop_probability, silence_loop_probability in parameters:

        exp_name = f"sub-{subsampling}_lm-cheating_transition-{lm_scale}_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}"
        if use_pruning:
            exp_name = exp_name + "_pruning"

        initializer_config = LateInitConfig()
        if initialization == "cheating":
            initializer_config = PreloadCentroidsInitializerConfig(centroids_path=input_data[input_data_key]["cheating_centroids"])
            exp_name = exp_name + "_cheating"
        if initialization == "random":
            initializer_config = StreamingStandardInitializerConfig(seed=42)
            exp_name = exp_name + "_random"


        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=None,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            use_tree_search=False,
            max_beam_size=20000 if use_pruning else None,
            score_threshold=10000.0 if use_pruning else None,
            lm_order=2,
            use_eow_phonemes=use_eow_phonemes,
            cheating=True,
            corpus=corpus_file
        )

        clustering_callback_config = ClusteringCallbackConfig(
            num_clusters=40 if not use_eow_phonemes else 79,
            initializer_config=initializer_config,
            recognition_config=recognition_config,
            lexicon_path=create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False),
            subsampling=subsampling,
            rasr_path=rasr_path,
            num_workers=15,
        )

        exp_result = clustering(
            num_epochs=num_epochs,
            sampled_segments=All,
            cluster_callback_config=clustering_callback_config,
            hdf_path=input_data[input_data_key]["features"],
            precomputed=True,
            log_verbosity=5,
            device="cpu",
        )

        tk.register_output(f"guided_kmeans/cheated_seg_plus_lm/statistics/{exp_name}.json", exp_result.out_statistics)


        for recog_epoch in range(num_epochs+1):     # run recognition after each epoch to see how PER develops

            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch] if recog_epoch > 0 else input_data[input_data_key]["cheating_centroids"],
                recog_rasr_config=recognition_config_decode,
                distance_scale=1.0,
                subsampling=subsampling,
            )

            res = decode_and_score(
                exp_name + f"_epoch-{recog_epoch}",
                "train-clean-100",
                decode_config,
                dataset_config,
                rasr_path=rasr_path,
                device="cpu",
            )
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per)
            recog_results.append(res)

            res_cv = decode_and_score(
                exp_name + f"_cv_epoch-{recog_epoch}", "cv",
                decode_config,
                cv_dataset_config,
                rasr_path=rasr_path,
                device="cpu",
                corpus_key="train-other-960",
            )
            tk.register_output(f"guided_kmeans/cheated_seg_plus_lm/recognition/{exp_name}_cv_epoch-{recog_epoch}_per", res_cv.per)
            recog_results_cv.append(res_cv)

    tk.register_report(f"guided_kmeans/cheated_seg_plus_lm/recognition/report_{input_data_key}.txt", values=create_report(recog_results), required=True)
    tk.register_report(f"guided_kmeans/cheated_seg_plus_lm/recognition/report_cv.txt", values=create_report(recog_results_cv), required=True)


def py():
    run()
