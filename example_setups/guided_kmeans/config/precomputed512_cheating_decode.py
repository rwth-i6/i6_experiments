from sisyphus import tk, Job, Task

from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import create_recog_rasr_config, create_lexicon
from i6_experiments.example_setups.guided_kmeans.setup.phoneme_frequency import get_sampled_segments_file
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import decode_and_score, DecodeConfig
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import DatasetConfig, RandomNumber, All,  SegmentFile
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import RecogConfig

verbosity = 1
rasr_path = tk.Path("/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard")

centroids_path = tk.Path("/u/mann/experiments/2026-06-09--guided-k-means/test/cheating_centroids/centroids.npy")

features_path = [
    tk.Path(
        f"/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame/pca512_frame/train_hdf/train.{i:03d}.hdf",
        # "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/example_setups/librispeech/phmm/fairseq_feature_jobs/FairseqNpyFeaturesToReturnnHDFJob.SmIsELgZKt0N/output/train_pca512_frame.{i:02d}.hdf"
    )
    for i in range(1, 21)
]

parameters = [
    # subsampling, lm_scale, label_loop_prob, sil_loop_prob
    (3, lm_scale, 0.1, 0.1)
    for lm_scale in [1.0, 5.0, 10.0, 15.0]
] + [
    # subsampling, lm_scale, label_loop_prob, sil_loop_prob
    (None, lm_scale, loop_prob, loop_prob)
    for lm_scale in [1.0, 5.0, 10.0, 15.0]
    for loop_prob in [0.05, 0.1, 0.2, 0.3, 0.5]
] + [
    (ss, 0.0, 0.5, 0.5)
    for ss in [3, None]
]

parameters = [
    (None, 10, loop, loop)
    for loop in [0.1, 0.3, 0.5, 0.7]
]

def py():
    recog_results = []

    for subsampling, lm_scale, loop_probability, silence_loop_probability in parameters:
        exp_name = f"sub-{subsampling}-lm-{lm_scale}_loop-{loop_probability}-sil-loop-{silence_loop_probability}"

        recognition_config = create_recog_rasr_config(
            lm_scale=lm_scale,
            loop_probability=loop_probability,
            silence_loop_probability=silence_loop_probability,
            use_tree_search=False,
        )

        dataset_config = DatasetConfig(
            audio_hdf_path=features_path,
            sampling_method=RandomNumber(100),
            #sampling_method=SegmentFile(get_sampled_segments_file(min_phoneme_count=5)),
            precomputed=True,
        )

        decode_config = DecodeConfig(
            centroids=centroids_path,
            recog_rasr_config=recognition_config,
            distance_scale=1.0,
            subsampling=subsampling,
        )

        res = decode_and_score(
            exp_name,
            # "train-clean-100-dbg",
            corpus_name="train-other-960",
            config=decode_config,
            dataset_config=dataset_config,
            rasr_path=rasr_path,
        )
        tk.register_output(f"guided_kmeans/recognition/test/cheating_centroids/{exp_name}_per", res.per)
        recog_results.append(res)

    # tk.register_report(f"guided_kmeans/recognition/test/cheating_centroids/report.txt", values=create_report(recog_results), required=True)

    return recog_results
