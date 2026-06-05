from sisyphus import tk

from ...segment_clustering_jobs import MergeConsecutiveClusterIdsJob


def eow_phon_phmm_ls960_wav2vec2_large_layer15_pca_merge_cluster_ids():
    prefix_name = (
        "example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca_merge_cluster_ids"
    )
    source_prefix = (
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_wav2vec2_large_layer15_pca/segment_clustering/"
        "layer15_pca512_peak_prom0p02_full_train_k128_4m"
    )

    cluster_hdfs = [tk.Path(f"{source_prefix}/cluster_labels_k128.{idx:03d}.hdf") for idx in range(10)]
    merge_job = MergeConsecutiveClusterIdsJob(
        cluster_hdfs,
        output_filename="cluster_labels_k128_merged.hdf",
        report_filename="cluster_labels_k128_merged_report.txt",
        mem_rqmt=16,
        time_rqmt=8,
    )
    merge_job.add_alias(prefix_name + "/k128_peak_prom0p02/merge_all_parts")
    tk.register_output(
        prefix_name + "/k128_peak_prom0p02/cluster_labels_k128_merged.hdf",
        merge_job.out_hdf,
    )
    tk.register_output(
        prefix_name + "/k128_peak_prom0p02/cluster_labels_k128_merged_report.txt",
        merge_job.out_report,
    )


py = eow_phon_phmm_ls960_wav2vec2_large_layer15_pca_merge_cluster_ids
