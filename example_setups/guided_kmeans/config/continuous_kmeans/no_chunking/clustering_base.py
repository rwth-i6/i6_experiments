from sisyphus import tk

import sys

from i6_experiments.example_setups.guided_kmeans.setup.clustering_config import (
    clustering,
    ClusteringCallbackConfig,
    LateInitConfig,
    StreamingStandardInitializerConfig,
    KMeansPlusPlusReservoirInitializerConfig,
    PickleCheatingCentroidInitializerConfig,
    PreloadCentroidsInitializerConfig,
)

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config, create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, RandomNumber, All, SegmentFile
from i6_experiments.example_setups.guided_kmeans.setup.centroid_metrics import CentroidCosineSimilarityJob, PhonemeL1DistanceJob, AverageTotalScoreJob, AverageNamedScoreJob
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as _BASE_INPUT_DATA,
    CHEATING_CENTROIDS_DBG,
    FEATURES_LS_CV as _CV_FEATURES,
    FEATURES_LS100H_SEGMENTED,
    SEGMENTS_LS_CV as _CV_SEGMENTS,
    GMM_ALIGNMENT_DBG as _GMM_ALIGNMENT_DBG,
    PHONEME_FREQUENCIES_LS100H,
)
from i6_experiments.example_setups.guided_kmeans import tools

# train-clean-100-dbg uses the DBG-subset centroids in this config
input_data = {
    **_BASE_INPUT_DATA,
    "train-clean-100-dbg": {
        **_BASE_INPUT_DATA["train-clean-100-dbg"],
        "cheating_centroids": CHEATING_CENTROIDS_DBG,
    },
}


def _test_forward_backward(
    use_eow_phonemes: bool = False,
    input_data_key: str = "ls-100",
    num_epochs: int = 2,
    lm_order: int = 7,
    lm_scale: float = 2000.0,
    transition_scale: float | None = None,
    loop_probability: float = 0.7,
    silence_loop_probability: float = 0.7,
    use_pruning: bool = True,
    distance_scale: float = 1.0,
):
    ts_display = transition_scale if transition_scale is not None else lm_scale
    exp_name = (
        f"fb_lm-{lm_order}gram-{lm_scale}-transition-{ts_display}"
        f"_loop-{loop_probability}-sil-loop-{silence_loop_probability}"
        f"_{input_data_key}_cheating"
    )

    recognition_config = create_recog_rasr_config(
        lm_scale=lm_scale,
        emission_scale=1.0,
        transition_scale=transition_scale,
        loop_probability=loop_probability,
        silence_loop_probability=silence_loop_probability,
        lm_order=lm_order,
        use_eow_phonemes=use_eow_phonemes,
        use_forward_backward_search=True,
        max_beam_size=1000 if use_pruning else None,
        score_threshold=5.0 if use_pruning else None,
        rasr_binary_path=tools.RASR_PATH_FORWARD_BACKWARD,
    )

    clustering_callback_config = ClusteringCallbackConfig(
        num_clusters=40 if not use_eow_phonemes else 79,
        initializer_config=PreloadCentroidsInitializerConfig(
            centroids_path=input_data[input_data_key]["cheating_centroids"]
        ),
        recognition_config=recognition_config,
        lexicon_path=create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False),
        subsampling=None,
        distance_scale=distance_scale,
        rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
        num_workers=1,
        use_forward_backward=True,
    )

    exp_result = clustering(
        num_epochs=num_epochs,
        sampled_segments=All,
        cluster_callback_config=clustering_callback_config,
        hdf_path=input_data[input_data_key].get("features_sharded", input_data[input_data_key]["features"]),
        partition_epoch=10 if "features_sharded" in input_data[input_data_key] else 1,
        precomputed=True,
        log_verbosity=5,
    )

    tk.register_output(f"guided_kmeans/testing_experimental/results/forward_backward_test/cheating/statistics/{exp_name}.json", exp_result.out_statistics)

    cv_dataset_config = DatasetConfig(
        audio_hdf_path=_CV_FEATURES,
        sampling_method=SegmentFile(_CV_SEGMENTS),
        precomputed=True,
    )

    recog_results = []
    for lm_scale_recog in [2000.0, 5000.0]:
        for recog_epoch in range(num_epochs + 1):
            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                recog_rasr_config=create_recog_rasr_config(
                    lm_scale=lm_scale_recog,
                    emission_scale=1.0,
                    transition_scale=lm_scale_recog,
                    loop_probability=loop_probability,
                    silence_loop_probability=silence_loop_probability,
                    lm_order=lm_order,
                    use_eow_phonemes=use_eow_phonemes,
                ),
                distance_scale=1.0,
                subsampling=None,
                write_frame_labels=True,
                num_workers=1,
            )

            res = decode_and_score(
                exp_name + f"_epoch-{recog_epoch}_scale-{lm_scale_recog}",
                "train-clean-100-dbg",
                decode_config,
                dataset_config,
                rasr_path=tools.RASR_PATH,
                corpus_key="train-clean-100",
            )
            #tk.register_output(f"guided_kmeans/fb_test/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per)
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_fb_epoch-{recog_epoch}_per", res.per)
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_fb_epoch-{recog_epoch}_confusion", res.confusion_pairs)

            res_cv = decode_and_score(exp_name + f"_cv_epoch-{recog_epoch}_scale-{lm_scale_recog}", "cv", decode_config, cv_dataset_config, rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960")
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_fb_cv_epoch-{recog_epoch}_scale-{lm_scale_recog}_per", res_cv.per)

            if res.frame_labels is not None:
                fer_job = FrameErrorRateJob(res.frame_labels, _GMM_ALIGNMENT_DBG, create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False))
                res.fer = fer_job.out_fer
                res.frame_confusion_pairs = fer_job.out_frame_confusion_pairs
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_fer", res.fer)
                tk.register_output(f"guided_kmeans/testing_experimental/confusion_pairs/{exp_name}_epoch-{recog_epoch}_frame_confusion", res.frame_confusion_pairs)
            res.mean_cos_sim = CentroidCosineSimilarityJob(exp_result.out_centroids[recog_epoch]).out_mean_cos_sim
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_cos_sim", res.mean_cos_sim)
            if recog_epoch < num_epochs:
                res.l1_dist = PhonemeL1DistanceJob(exp_result.out_statistics, recog_epoch, PHONEME_FREQUENCIES_LS100H).out_l1_dist
                res.avg_total_score = AverageTotalScoreJob(exp_result.out_statistics, recog_epoch).out_avg_total_score
                res.avg_am_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_am_score").out_avg_score
                res.avg_transition_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_transition_score").out_avg_score
                res.avg_lm_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_lm_score").out_avg_score
                res.avg_segment_duration = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_segment_duration").out_avg_score
                #tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_l1", res.l1_dist)
                #tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_score", res.avg_total_score)
                #tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_am_score", res.avg_am_score)
                #tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_transition_score", res.avg_transition_score)
                #tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_lm_score", res.avg_lm_score)
                #tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_avg_segment_duration", res.avg_segment_duration)
            recog_results.append(res)

    return recog_results


def _run_test_experiments(
    parameters,
    input_data_key: str,
    use_eow_phonemes: bool,
    num_epochs: int,
    use_pruning: bool,
    initialization: str,
) -> list:
    recog_results = []
    for subsampling, lm_order, lm_scale, transition_scale, loop_probability, silence_loop_probability in parameters:
        ts_display = transition_scale if transition_scale is not None else lm_scale
        exp_name = f"sub-{subsampling}_lm-{lm_order}gram-{lm_scale}-transition-{ts_display}_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}"
        if use_pruning:
            exp_name = exp_name + "_pruning"

        initializer_config = LateInitConfig()
        if initialization == "cheating":
            initializer_config = PreloadCentroidsInitializerConfig(centroids_path=input_data[input_data_key]["cheating_centroids"])
            exp_name = exp_name + "_cheating"
        if initialization == "random":
            initializer_config = StreamingStandardInitializerConfig(seed=42)
            exp_name = exp_name + "_random"
        if initialization == "kmeansapp":
            initializer_config = KMeansPlusPlusReservoirInitializerConfig(seed=42)
            exp_name = exp_name + "_kmeanspp"

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=transition_scale,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            use_tree_search=False,
            max_beam_size=5000 if use_pruning else None,
            score_threshold=50000.0 if use_pruning else None,
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
            num_workers=7
        )

        exp_result = clustering(
            num_epochs=num_epochs,
            sampled_segments=All,
            cluster_callback_config=clustering_callback_config,
            hdf_path=input_data[input_data_key]["features"],
            partition_epoch=1,
            precomputed=True,
            log_verbosity=3,
        )

        tk.register_output(f"guided_kmeans/testing_experimental/statistics/{exp_name}.json", exp_result.out_statistics)

        cv_dataset_config = DatasetConfig(
            audio_hdf_path=_CV_FEATURES,
            sampling_method=SegmentFile(_CV_SEGMENTS),
            precomputed=True,
        )

        for recog_epoch in range(num_epochs+1):     # run recognition after each epoch to see how PER develops

            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                recog_rasr_config=recognition_config,
                distance_scale=1.0,
                subsampling=subsampling,
            )

            res = decode_and_score(exp_name + f"_epoch-{recog_epoch}", "train-clean-100-dbg", decode_config, dataset_config, rasr_path=tools.RASR_PATH, corpus_key="train-clean-100")
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per)
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_confusion", res.confusion_pairs)

            res_cv = decode_and_score(exp_name + f"_cv_epoch-{recog_epoch}", "cv", decode_config, cv_dataset_config, rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960")
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_cv_epoch-{recog_epoch}_per", res_cv.per)
            if res.frame_labels is not None:
                fer_job = FrameErrorRateJob(res.frame_labels, _GMM_ALIGNMENT_DBG, create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False))
                res.fer = fer_job.out_fer
                res.frame_confusion_pairs = fer_job.out_frame_confusion_pairs

            res.mean_cos_sim = CentroidCosineSimilarityJob(exp_result.out_centroids[recog_epoch]).out_mean_cos_sim
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_cos_sim", res.mean_cos_sim)
            if recog_epoch < num_epochs:
                res.l1_dist = PhonemeL1DistanceJob(exp_result.out_statistics, recog_epoch, PHONEME_FREQUENCIES_LS100H).out_l1_dist
                res.avg_total_score = AverageTotalScoreJob(exp_result.out_statistics, recog_epoch).out_avg_total_score
                res.avg_am_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_am_score").out_avg_score
                res.avg_transition_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_transition_score").out_avg_score
                res.avg_lm_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_lm_score").out_avg_score
                res.avg_segment_duration = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_segment_duration").out_avg_score
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_l1", res.l1_dist)
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_score", res.avg_total_score)
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_am_score", res.avg_am_score)
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_transition_score", res.avg_transition_score)
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_lm_score", res.avg_lm_score)
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_avg_segment_duration", res.avg_segment_duration)

            recog_results.append(res)

    return recog_results


def test():

    use_eow_phonemes = False
    num_epochs = 1
    use_pruning = False

    input_data_key = "ls-100"
    initialization = "cheating"

    # (subsampling, lm_order, lm_scale, transition_scale, loop_probability, silence_loop_probability)
    parameters = [
        (None, 3, lm_scale, transition_scale, loop_prob, loop_prob)
        for lm_scale in [500.0, 2000.0, 5000.0, 10000.0, 50000.0]
        for transition_scale in [50.0, 200.0, 500.0, 2000.0, 5000.0, 10000.0, 50000.0]
        for loop_prob in [0.1, 0.4, 0.7, 0.9]
    ]

    kwargs = dict(
        input_data_key=input_data_key,
        use_eow_phonemes=use_eow_phonemes,
        num_epochs=num_epochs,
        use_pruning=use_pruning,
        initialization=initialization,
    )

    recog_results = _run_test_experiments(parameters, **kwargs)

    tk.register_report(f"guided_kmeans/testing_experimental/results/{initialization}/report_{input_data_key}.txt", values=create_report(recog_results), required=True)


# Forward-backward soft-assignment experiment
def fb_test():

    use_eow_phonemes = False
    num_epochs = 1
    use_pruning = False

    input_data_key = "ls-100"
    initialization = "cheating"

    # (subsampling, lm_order, lm_scale, transition_scale, loop_probability, silence_loop_probability)
    parameters = [
        (None, 3, lm_scale, lm_scale, loop_prob, loop_prob)
        for lm_scale in [0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0]
        #for transition_scale in [0.3, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0]
        for loop_prob in [0.4, 0.7]
    ]

    recog_results = []

    for subsampling, lm_order, lm_scale, transition_scale, loop_probability, silence_loop_probability in parameters:

        recog_results += _test_forward_backward(
            use_eow_phonemes=use_eow_phonemes,
            input_data_key=input_data_key,
            num_epochs=num_epochs,
            lm_order=lm_order,
            lm_scale=lm_scale,
            transition_scale=transition_scale,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            use_pruning=True,
            distance_scale=0.0001,
        )

    tk.register_report(f"guided_kmeans/testing_experimental/results/forward-backward_test/{initialization}/results/report_{input_data_key}_forward-backward_test.txt", values=create_report(recog_results), required=True)



def cheating_segmentation():

    use_eow_phonemes = False
    num_epochs = 1
    use_pruning = False

    input_data_key = "ls-100"
    initialization = "cheating"

    # (subsampling, lm_order, lm_scale, transition_scale, loop_probability, silence_loop_probability)
    parameters = [
        (None, 3, lm_scale, None, 0.0, 0.0) for lm_scale in [500.0, 2000.0, 5000.0, 8000.0, 10000.0]
    ]

    recog_results = []

    for subsampling, lm_order, lm_scale, transition_scale, loop_probability, silence_loop_probability in parameters:

        ts_display = transition_scale if transition_scale is not None else lm_scale
        exp_name = f"sub-{subsampling}_lm-{lm_order}gram-{lm_scale}-transition-{ts_display}_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}"
        if use_pruning:
            exp_name = exp_name + "_pruning"

        initializer_config = LateInitConfig()
        if initialization == "cheating":
            initializer_config = PreloadCentroidsInitializerConfig(centroids_path=input_data[input_data_key]["cheating_centroids"])
            #initializer_config = PreloadCentroidsInitializerConfig(centroids_path=tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids_segmented.npy"))
            exp_name = exp_name + "_cheating"
        if initialization == "random":
            initializer_config = StreamingStandardInitializerConfig(seed=42)
            exp_name = exp_name + "_random"
        if initialization == "kmeansapp":
            initializer_config = KMeansPlusPlusReservoirInitializerConfig(seed=42)
            exp_name = exp_name + "_kmeanspp"

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=transition_scale,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            use_tree_search=False,
            max_beam_size=20000 if use_pruning else None,
            score_threshold=40000 if use_pruning else None,
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
            num_workers=7,
        )

        exp_result = clustering(
            num_epochs=num_epochs,
            sampled_segments=All,
            cluster_callback_config=clustering_callback_config,
            hdf_path=FEATURES_LS100H_SEGMENTED,
            precomputed=True,
            log_verbosity=3,
        )

        tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/statistics/{exp_name}.json", exp_result.out_statistics)

        cv_dataset_config = DatasetConfig(
            audio_hdf_path=_CV_FEATURES,
            sampling_method=SegmentFile(_CV_SEGMENTS),
            precomputed=True,
        )

        recognition_config_decode = create_recog_rasr_config(
            lm_scale=lm_scale,
            emission_scale=1.0,
            transition_scale=transition_scale,
            loop_probability=0.7,
            silence_loop_probability=0.7,
            use_tree_search=False,
            max_beam_size=20000 if use_pruning else None,
            score_threshold=40000 if use_pruning else None,
            lm_order=lm_order,
            use_eow_phonemes=use_eow_phonemes
        )

        for recog_epoch in range(num_epochs+1):     # run recognition after each epoch to see how PER develops

            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                recog_rasr_config=recognition_config_decode,
                distance_scale=1.0,
                subsampling=subsampling,
                write_frame_labels=True,
                num_workers=1,
            )

            res = decode_and_score(exp_name + f"_epoch-{recog_epoch}", "train-clean-100-dbg", decode_config, dataset_config, rasr_path=tools.RASR_PATH, corpus_key="train-clean-100")
            tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per)
            tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/confusion_pairs/{exp_name}_epoch-{recog_epoch}_confusion", res.confusion_pairs)

            res_cv = decode_and_score(exp_name + f"_cv_epoch-{recog_epoch}", "cv", decode_config, cv_dataset_config, rasr_path=tools.RASR_PATH, device="cpu", corpus_key="train-other-960")
            tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_cv_epoch-{recog_epoch}_per", res_cv.per)
            if res.frame_labels is not None:
                #tk.register_output(f"guided_kmeans/cheating_segmentation/frame_labels/{exp_name}_epoch-{recog_epoch}", res.frame_labels)
                fer_job = FrameErrorRateJob(res.frame_labels, _GMM_ALIGNMENT_DBG, create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False))
                res.fer = fer_job.out_fer
                res.frame_confusion_pairs = fer_job.out_frame_confusion_pairs
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_fer", res.fer)
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/confusion_pairs/{exp_name}_epoch-{recog_epoch}_frame_confusion", res.frame_confusion_pairs)

            res.mean_cos_sim = CentroidCosineSimilarityJob(exp_result.out_centroids[recog_epoch]).out_mean_cos_sim
            tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_cos_sim", res.mean_cos_sim)
            if recog_epoch < num_epochs:
                res.l1_dist = PhonemeL1DistanceJob(exp_result.out_statistics, recog_epoch, PHONEME_FREQUENCIES_LS100H).out_l1_dist
                res.avg_total_score = AverageTotalScoreJob(exp_result.out_statistics, recog_epoch).out_avg_total_score
                res.avg_am_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_am_score").out_avg_score
                res.avg_transition_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_transition_score").out_avg_score
                res.avg_lm_score = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_lm_score").out_avg_score
                res.avg_segment_duration = AverageNamedScoreJob(exp_result.out_statistics, recog_epoch, "average_segment_duration").out_avg_score
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_l1", res.l1_dist)
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_score", res.avg_total_score)
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_am_score", res.avg_am_score)
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_transition_score", res.avg_transition_score)
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_lm_score", res.avg_lm_score)
                tk.register_output(f"guided_kmeans/testing_experimental/results/cheating_segmentation/recognition/{exp_name}_epoch-{recog_epoch}_avg_segment_duration", res.avg_segment_duration)

            recog_results.append(res)

    tk.register_report(f"guided_kmeans/testing_experimental/results/cheating_segmentation/results/{initialization}/report_{input_data_key}.txt", values=create_report(recog_results), required=True)



def py():
    test()