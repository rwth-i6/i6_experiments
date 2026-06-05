from sisyphus import tk

from ...cluster_eval_jobs import ComputeClusterPurityPnmiJob


def eow_phon_phmm_ls960_wav2vec2_large_layer15_pca_cluster_eval():
    prefix_name = (
        "example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca_cluster_eval"
    )

    pca_prefix = "output/example_setups/librispeech/phmm_standalone_2024/ls960_wav2vec2_large_layer15_pca"
    alignment_hdfs = [tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf") for i in range(1, 201)]
    utterance_select = tk.Path("/work/asr4/zyang/corpora/librispeech/960/mlm_splits/dev1.indices.txt")
    p02_start_hdf = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/segmenter_jobs/DetectPeaksFromScoreHDFJob.Bf3WdJsaZWqv/"
        "output/peak_segment_starts.hdf"
    )
    pca512_features = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/feature_pca_jobs/ApplyPcaToFeatureHDFJob.JfBOA9zHcXBA/"
        "output/features_layer15_pca512.hdf"
    )
    gmm_idx_to_phoneme = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/misc/dump_train_other_960_eow_alignment_hdf/"
        "GenerateEowMonophoneStateTyingJob.LcucMA7avmoo/output/idx_to_phoneme.txt"
    )

    for num_clusters in (60, 40, 128):
        cluster_hdfs = [
            tk.Path(
                f"{pca_prefix}/segment_clustering/"
                f"layer15_pca512_peak_prom0p02_full_train_k{num_clusters}_4m/"
                f"cluster_labels_k{num_clusters}.{part_idx:03d}.hdf"
            )
            for part_idx in range(10)
        ]
        job = ComputeClusterPurityPnmiJob(
            alignment_hdfs=alignment_hdfs,
            cluster_hdfs=cluster_hdfs,
            utterance_select=utterance_select,
            idx_to_phoneme=gmm_idx_to_phoneme,
            strip_eow_from_alignment=True,
            compute_frame_error_rate=True,
            cluster_input_type="segment",
            segment_start_hdf=p02_start_hdf,
            feature_hdf=pca512_features,
            segment_downsample_rate=2,
            alignment_downsample_rate=2,
            segment_random_seed=1,
            num_clusters=num_clusters,
            store_joint_probability=True,
            joint_probability_filename=f"phone_cluster_joint_probability_k{num_clusters}_p02_1pct.npz",
            output_filename=f"cluster_purity_pnmi_k{num_clusters}_p02_1pct.txt",
            mem_rqmt=24,
            time_rqmt=5,
        )
        job.add_alias(prefix_name + f"/p02_segment_k{num_clusters}_1pct")
        tk.register_output(prefix_name + f"/p02_segment_k{num_clusters}_1pct.txt", job.out_report)
        tk.register_output(prefix_name + f"/p02_segment_k{num_clusters}_1pct_joint_probability.npz", job.out_joint_probability)


py = eow_phon_phmm_ls960_wav2vec2_large_layer15_pca_cluster_eval
