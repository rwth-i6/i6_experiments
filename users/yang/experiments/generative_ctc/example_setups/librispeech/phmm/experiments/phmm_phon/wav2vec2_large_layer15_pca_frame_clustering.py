from sisyphus import tk

from ...cluster_eval_jobs import ComputeClusterPurityPnmiJob
from ...frame_clustering_jobs import DetectClusterChangeStartFramesJob, create_faiss_frame_clustering_pipeline
from ...segmenter_jobs import CompareSegmentStartHDFJob, DumpGmmAlignmentSegmentStartsJob


def eow_phon_phmm_ls960_wav2vec2_large_layer15_pca_frame_clustering():
    prefix_name = (
        "example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca_frame_clustering"
    )

    pca512_features = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/feature_pca_jobs/ApplyPcaToFeatureHDFJob.JfBOA9zHcXBA/"
        "output/features_layer15_pca512.hdf"
    )

    clustering_outputs = create_faiss_frame_clustering_pipeline(
        output_prefix=prefix_name + "/layer15_pca512_full_train_k128_4m",
        feature_hdfs=[pca512_features],
        num_clusters=128,
        num_samples=4_000_000,
        random_seed=1,
        use_cache_manager=False,
        train_gpu_mem=24,
        assign_gpu_mem=24,
    )

    alignment_hdfs = [tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf") for i in range(1, 201)]
    utterance_select = tk.Path("/work/asr4/zyang/corpora/librispeech/960/mlm_splits/dev1.indices.txt")
    gmm_idx_to_phoneme = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/misc/dump_train_other_960_eow_alignment_hdf/"
        "GenerateEowMonophoneStateTyingJob.LcucMA7avmoo/output/idx_to_phoneme.txt"
    )

    eval_job = ComputeClusterPurityPnmiJob(
        alignment_hdfs=alignment_hdfs,
        cluster_hdfs=clustering_outputs["assignments"],
        utterance_select=utterance_select,
        idx_to_phoneme=gmm_idx_to_phoneme,
        strip_eow_from_alignment=True,
        compute_frame_error_rate=True,
        cluster_input_type="frame",
        alignment_downsample_rate=2,
        num_clusters=128,
        store_joint_probability=True,
        joint_probability_filename="phone_cluster_joint_probability_frame_k128_1pct.npz",
        output_filename="cluster_purity_pnmi_frame_k128_1pct.txt",
        mem_rqmt=24,
        time_rqmt=5,
    )
    eval_job.add_alias(prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_eval_1pct_no_eow")
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_purity_pnmi_k128_1pct_no_eow.txt",
        eval_job.out_report,
    )
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_joint_probability_k128_1pct_no_eow.npz",
        eval_job.out_joint_probability,
    )

    k128_start_job = DetectClusterChangeStartFramesJob(
        clustering_outputs["assignments"],
        output_filename="frame_cluster_change_starts_20ms.hdf",
        output_scaled_filename="frame_cluster_change_starts_10ms.hdf",
        output_segment_cluster_filename="frame_cluster_change_segment_clusters.hdf",
        scaled_frame_factor=2,
    )
    k128_start_job.add_alias(prefix_name + "/layer15_pca512_full_train_k128_4m/cluster_change_starts")
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_change_starts_20ms.hdf",
        k128_start_job.out_hdf,
    )
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_change_starts_10ms.hdf",
        k128_start_job.out_scaled_hdf,
    )
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_change_segment_clusters.hdf",
        k128_start_job.out_segment_cluster_hdf,
    )
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_change_start_stats.txt",
        k128_start_job.out_stats,
    )

    gmm_start_job = DumpGmmAlignmentSegmentStartsJob(alignment_hdfs)
    gmm_start_job.add_alias(prefix_name + "/gmm_alignment_segment_starts")
    tk.register_output(prefix_name + "/gmm_alignment_segment_starts.hdf", gmm_start_job.out_hdf)

    compare_job = CompareSegmentStartHDFJob(
        source_hdf=k128_start_job.out_scaled_hdf,
        target_hdf=gmm_start_job.out_hdf,
        tolerance=2,
        ignore_zero=True,
        target_alignment_hdfs=alignment_hdfs,
        output_filename="frame_cluster_k128_vs_gmm_boundary_comparison_10ms.txt",
    )
    compare_job.add_alias(prefix_name + "/layer15_pca512_full_train_k128_4m/compare_cluster_change_starts_10ms_to_gmm")
    tk.register_output(
        prefix_name + "/layer15_pca512_full_train_k128_4m/frame_cluster_k128_vs_gmm_boundary_comparison_10ms.txt",
        compare_job.out_report,
    )


py = eow_phon_phmm_ls960_wav2vec2_large_layer15_pca_frame_clustering
