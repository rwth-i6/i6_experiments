from sisyphus import tk, Job, Task

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, RandomNumber, All,  SegmentFile
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import RecogConfig
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from .. import tools

parameters = \
    [
        (2, 1, lm_scale, 0.1, 0.1) for lm_scale in [5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0]
    ] + \
    [
        (3, 1, lm_scale, 0.1, 0.1) for lm_scale in [5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0]
    ] + \
    [
        (2, 2, lm_scale, 0.1, 0.1) for lm_scale in [5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0]
    ] + \
    [
        (3, 2, lm_scale, 0.1, 0.1) for lm_scale in [5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0]
    ] + \
    [
        (2, 3, lm_scale, 0.1, 0.1) for lm_scale in [5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0]
    ] + \
    [
        (3, 3, lm_scale, 0.1, 0.1) for lm_scale in [5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0]
    ] + \
    [
        (3, 1, 2000.0, loop_prob, loop_prob) for loop_prob in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ] + \
    [
        (3, 2, 2000.0, loop_prob, loop_prob) for loop_prob in [0.02, 0.05, 0.1, 0.2, 0.3]
    ] + \
    [
        (3, 3, 2000.0, loop_prob, loop_prob) for loop_prob in [0.02, 0.05, 0.1, 0.2]
    ]

def py():
    recog_results = []

    for lm_order, subsampling, lm_scale, loop_probability, silence_loop_probability in parameters:
        exp_name = f"sub-{subsampling}-lm-{lm_order}gram-{lm_scale}_loop-{loop_probability}-sil-loop-{silence_loop_probability}"

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            lm_order=lm_order,
            use_tree_search=False,
        )

        dataset_config = DatasetConfig(
            audio_hdf_path=tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/filtered_features_train-clean-100-dbg.hdf"),
            sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
            precomputed=True,
        )

        decode_config = DecodeConfig(
            centroids=tk.Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/train-clean-100-dbg/centroids.npy"),
            recog_rasr_config=recognition_config,
            distance_scale=1.0,
            subsampling=subsampling,
        )

        res = decode_and_score(
            exp_name,
            "train-clean-100-dbg",
            config=decode_config,
            dataset_config=dataset_config,
            rasr_path=tools.RASR_PATH,
        )

        tk.register_output(f"guided_kmeans/cheating_centroids_decode_test/{exp_name}_per", res.per)

        recog_results.append(res)

    tk.register_report(f"guided_kmeans/cheating_centroids_decode_test/report.txt", values=create_report(recog_results), required=True)

    return recog_results
