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
from i6_experiments.example_setups.guided_kmeans.setup.centroid_metrics import CentroidCosineSimilarityJob, PhonemeL1DistanceJob, AverageTotalScoreJob

from i6_experiments.example_setups.guided_kmeans import tools

# TODO maybe use DistributedFileDataset

input_data = {
    "train-clean-100-dbg": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/filtered_features_train-clean-100-dbg.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/train-clean-100-dbg/centroids.npy"),
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/train-clean-100-dbg-segments.txt"),
    },
    "ls-100": {
        "features": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/wav2vec2_ls100h.hdf"),
        "cheating_centroids": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/centroids.npy"), # computed on the full 960h
        "segment_file": tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list/ls100h-segments.txt"),
    }
}

def run():

    use_eow_phonemes = False
    num_epochs = 5
    use_pruning = False

    input_data_key = "train-clean-100-dbg"
    initialization = "random"

    parameters = [
        (None, 4, 10000.0, 0.3, 0.3, None)
    ]

    recog_results = []

    for subsampling, lm_order, lm_scale, loop_probability, silence_loop_probability, lm_scale_schedule in parameters:

        decode_lm_scale_schedule = lm_scale_schedule

        exp_name = f"sub-{subsampling}_lm-{lm_order}gram-{lm_scale}_loop-{loop_probability}-sil-loop-{silence_loop_probability}_{input_data_key}_{lm_scale_schedule}"
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
            lm_scale_schedule=lm_scale_schedule,
            rasr_path=tools.RASR_PATH,
        )

        exp_result = clustering(
            num_epochs=num_epochs,
            sampled_segments=All,
            cluster_callback_config=clustering_callback_config,
            hdf_path=input_data[input_data_key]["features"],
            precomputed=True,
            log_verbosity=5,
        )

        tk.register_output(f"guided_kmeans/testing_experimental/statistics/{exp_name}.json", exp_result.out_statistics)


        for recog_epoch in range(num_epochs+1):     # run recognition after each epoch to see how PER develops

            dataset_config = DatasetConfig(
                audio_hdf_path=input_data["train-clean-100-dbg"]["features"],
                sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
                precomputed=True,
            )

            if decode_lm_scale_schedule is not None:
                decode_scale = decode_lm_scale_schedule[min(recog_epoch, len(decode_lm_scale_schedule) - 1)]
                decode_recognition_config = create_recog_rasr_config(
                    lm_scale=decode_scale,
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
            else:
                decode_recognition_config = recognition_config

            decode_config = DecodeConfig(
                centroids=exp_result.out_centroids[recog_epoch],
                recog_rasr_config=decode_recognition_config,
                distance_scale=1.0,
                subsampling=subsampling,
                write_frame_labels=True,
            )

            res = decode_and_score(exp_name + f"_epoch-{recog_epoch}", "train-clean-100-dbg", decode_config, dataset_config, rasr_path=tools.RASR_PATH)
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_per", res.per)
            if res.frame_labels is not None:
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_frame_labels", res.frame_labels)

            res.mean_cos_sim = CentroidCosineSimilarityJob(exp_result.out_centroids[recog_epoch]).out_mean_cos_sim
            tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_cos_sim", res.mean_cos_sim)
            if recog_epoch < num_epochs:
                res.l1_dist = PhonemeL1DistanceJob(exp_result.out_statistics, recog_epoch, tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/phoneme_frequencies/phoneme_frequencies_ls_100.txt")).out_l1_dist
                res.avg_total_score = AverageTotalScoreJob(exp_result.out_statistics, recog_epoch).out_avg_total_score
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_l1", res.l1_dist)
                tk.register_output(f"guided_kmeans/testing_experimental/recognition/{exp_name}_epoch-{recog_epoch}_score", res.avg_total_score)

            recog_results.append(res)

    tk.register_report(f"guided_kmeans/testing_experimental/recognition/{initialization}/report_{input_data_key}_analysis.txt", values=create_report(recog_results), required=True)

def py():
    test()