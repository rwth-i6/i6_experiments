from sisyphus import tk

import sys

from i6_experiments.example_setups.guided_kmeans.setup.clustering_config import (
    clustering,
    ClusteringCallbackConfig,
    LateInitConfig,
    StreamingStandardInitializerConfig,
    PickleCheatingCentroidInitializerConfig,
    PreloadCentroidsInitializerConfig,
    PreloadGMInitializerConfig
)

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config, create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import LatexTableReport, clustering_statistics
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, RandomNumber, All, SegmentFile

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    FEATURES_LS_CV as _CV_FEATURES,
    SEGMENTS_LS_CV as _CV_SEGMENTS,
)
from i6_experiments.example_setups.guided_kmeans import tools

def run():
    use_eow_phonemes = False
    num_epochs = 10
    use_pruning = False

    cpu = True
    num_workers = 15

    input_data_key = "ls-100"
    # input_data_key = "train-clean-100-dbg"
    initialization = "cheating"

    parameters = [
        (None, 3, lm, 0.2, 0.2)
        for lm in [20.0, 30.0, 40.0, 50.0]
    ] + [
        # (None, 3, lm, lp, lp)
        # for lm in [20.0, 30.0, 50.0]
        # for lp in [0.4, 0.6, 0.8]
    ]

    parameters = [
        (None, 3, lm, lp, lp)
        for lm in [30.0, 40.0, 50.0]
        for lp in [0.10, 0.15, 0.20, 0.25, 0.30]
    ]

    recog_results = []
    recog_results_cv = []

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=_CV_FEATURES,
        sampling_method=SegmentFile(_CV_SEGMENTS),
        precomputed=True,
    )

    # One block per (LM scale, loop probability), one line per decoded epoch. The
    # statistics columns come from the clustering job's epoch_statistics.json, the L1
    # distance from comparing its unigram phoneme frequencies against the transcription
    # priors in setup/constants.py.
    latex_report = LatexTableReport(
        columns=[
            "lm_scale", "loop_probability",
            "epoch", "per", "del", "ins", "sub",
            "l1", "am_score", "lm_score",
        ],
        sort_by=["lm_scale", "loop_probability"],
        caption="Guided k-means with covariances, cheating initialization, LibriSpeech 100h.",
    )
    # Same rows, cut down to the first and the last epoch. num_epochs-1 rather than
    # num_epochs because the final centroids are written after the last recognition
    # pass, so epoch num_epochs has no statistics of its own.
    latex_report_first_last = latex_report.view(
        epochs=(0, num_epochs - 1),
        caption=(
            "Guided k-means with covariances, cheating initialization, LibriSpeech 100h: "
            f"epoch 0 against epoch {num_epochs - 1}."
        ),
    )

    for subsampling, lm_order, lm_scale, loop_probability, silence_loop_probability in parameters:

        exp_name = f"sub-{subsampling}_lm-{lm_order}gram-{lm_scale}_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}"
        if use_pruning:
            exp_name = exp_name + "_pruning"

        initializer_config = LateInitConfig()
        assert initialization == "cheating"
        if initialization == "cheating":
            centroids_init = input_data[input_data_key]["cheating_centroids"]
            covs_init = input_data[input_data_key]["cheating_covs"]
            initializer_config = PreloadGMInitializerConfig(
                centroids_path=input_data[input_data_key]["cheating_centroids"],
                covs_path=input_data[input_data_key]["cheating_covs"]
            )
            exp_name = exp_name + "_cheating"
        elif initialization == "random":
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
            lm_order=lm_order,
            use_eow_phonemes=use_eow_phonemes
        )

        clustering_callback_config = ClusteringCallbackConfig(
            num_clusters=40 if not use_eow_phonemes else 79,
            initializer_config=initializer_config,
            recognition_config=recognition_config,
            lexicon_path=create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False),
            subsampling=subsampling,
            rasr_path=tools.RASR_PATH,
            num_workers=num_workers,
        )

        exp_result = clustering(
            num_epochs=num_epochs,
            sampled_segments=All,
            cluster_callback_config=clustering_callback_config,
            hdf_path=input_data[input_data_key]["features"],
            precomputed=True,
            log_verbosity=5,
            device="cpu" if cpu else "gpu"
        )

        # tk.register_output(f"guided_kmeans/testing_experimental/statistics/{exp_name}.json", exp_result.out_statistics)

        # once per experiment, not per epoch, so all epochs share the same jobs
        statistics = clustering_statistics(exp_result.out_statistics, name=exp_name)


        for recog_epoch in range(num_epochs+1):     # run recognition after each epoch to see how PER develops
        # for recog_epoch in range(1):     # run recognition after each epoch to see how PER develops

            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            assert exp_result.out_covs
            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch] if recog_epoch > 0 else centroids_init,
                recog_rasr_config=recognition_config,
                distance_scale=1.0,
                subsampling=subsampling,
                covs=exp_result.out_covs[recog_epoch] if recog_epoch > 0 else covs_init,
            )

            res = decode_and_score(
                exp_name + f"_epoch-{recog_epoch}",
                "train-clean-100-dbg",
                decode_config,
                dataset_config,
                rasr_path=tools.RASR_PATH,
                corpus_key="train-clean-100",
            )
            tk.register_output(f"guided_kmeans/cov/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per)
            recog_results.append(res)

            res_cv = decode_and_score(
                exp_name + f"_cv_epoch-{recog_epoch}",
                "cv",
                decode_config,
                cv_dataset_config,
                rasr_path=tools.RASR_PATH,
                device="cpu",
                corpus_key="train-other-960",
            )
            tk.register_output(f"guided_kmeans/cov/recognition/{exp_name}_cv_epoch-{recog_epoch}_per", res_cv.per)
            recog_results_cv.append(res_cv)

            latex_report.add_row(
                result=res,
                params={"lm_scale": lm_scale, "loop_probability": loop_probability},
                epoch=recog_epoch,
                statistics=statistics,
            )

    tk.register_report(f"guided_kmeans/cov/recognition/report_{input_data_key}.txt", values=create_report(recog_results), required=True)
    tk.register_report(f"guided_kmeans/cov/recognition/report_cv.txt", values=create_report(recog_results_cv), required=True)
    latex_report.register(f"guided_kmeans/cov/recognition/report_{input_data_key}.tex")
    latex_report_first_last.register(
        f"guided_kmeans/cov/recognition/report_{input_data_key}_first_last.tex"
    )


def py():
    run()
