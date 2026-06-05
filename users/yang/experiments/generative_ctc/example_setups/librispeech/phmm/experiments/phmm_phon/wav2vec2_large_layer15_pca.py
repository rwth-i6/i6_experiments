from dataclasses import asdict

from sisyphus import tk

from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datasets import MetaDataset, OggZipDataset

from ...data.phmm_common import DatasetSettings, get_audio_raw_datastream
from ...feature_pca_jobs import ApplyPcaToFeatureHDFJob
from ...phmm_config import get_forward_config
from ...phmm_default_tools import LIBRASR_WHEEL, MINI_RETURNN_ROOT, RETURNN_EXE
from ...phmm_rasr import CreateLibrasrVenvJob
from ...pytorch_networks.phmm.wav2vec2_hf_feature_extractor_cfg import ModelConfig
from ...segment_clustering_jobs import ComputeClusterToGmmPerJob
from ...segment_clustering_jobs import create_faiss_segment_clustering_pipeline
from ...segmenter_jobs import create_partitioned_segment_representation_jobs


def _build_existing_train_audio_dataset(settings: DatasetSettings):
    train_ogg = tk.Path("/u/zyang/setups/mini/work/i6_core/returnn/oggzip/BlissToOggZipJob.mNyFEqof29Q5/output/out.ogg.zip")
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)
    zip_dataset = OggZipDataset(
        files=[train_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=None,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )
    return MetaDataset(
        data_map={"raw_audio": ("zip_dataset", "data")},
        datasets={"zip_dataset": zip_dataset},
        seq_order_control_dataset="zip_dataset",
    )


def eow_phon_phmm_ls960_wav2vec2_large_layer15_pca():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_wav2vec2_large_layer15_pca"

    dataset_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=1,
        train_seq_ordering="sorted_reverse",
    )
    forward_dataset = _build_existing_train_audio_dataset(dataset_settings)

    returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja", "transformers"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    model_config = ModelConfig(
        hf_model_name="facebook/wav2vec2-large",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=False,
        return_layer=15,
    )

    feature_filename = "features_layer15.hdf"
    pca_state_filename = "pca_state_layer15_dim512.pt"
    pca_feature_filename = "features_layer15_pca512.hdf"

    forward_config = get_forward_config(
        network_module="phmm.wav2vec2_hf_feature_extractor",
        config={
            "forward": forward_dataset.as_returnn_opts(),
            "batch_size": 200 * 16000,
            "max_seqs": 80,
            "num_workers_per_gpu": 2,
            "torch_amp_options": {"dtype": "bfloat16"},
        },
        net_args={"model_config_dict": asdict(model_config)},
        decoder="phmm.wav2vec2_feature_pca_forward",
        decoder_args={
            "config": {
                "feature_output_filename": feature_filename,
                "pca_state_filename": pca_state_filename,
                "feature_dim": 1024,
                "pca_dim": 512,
                "covariance_chunk_size": 8192,
                "return_layer": 15,
            }
        },
    )

    forward_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=forward_config,
        log_verbosity=5,
        mem_rqmt=32,
        time_rqmt=168,
        device="gpu",
        cpu_rqmt=4,
        output_files=[feature_filename, pca_state_filename],
        returnn_python_exe=returnn_exe,
        returnn_root=MINI_RETURNN_ROOT,
    )
    forward_job.rqmt["gpu_mem"] = 48
    forward_job.add_alias(prefix_name + "/dump_layer15_features_and_fit_pca")
    tk.register_output(prefix_name + f"/{feature_filename}", forward_job.out_files[feature_filename])
    tk.register_output(prefix_name + f"/{pca_state_filename}", forward_job.out_files[pca_state_filename])

    apply_pca_job = ApplyPcaToFeatureHDFJob(
        feature_hdf=forward_job.out_files[feature_filename],
        pca_state=forward_job.out_files[pca_state_filename],
        output_filename=pca_feature_filename,
        chunk_size=100_000,
    )
    apply_pca_job.add_alias(prefix_name + "/apply_pca_to_layer15_features")
    tk.register_output(prefix_name + f"/{pca_feature_filename}", apply_pca_job.out_hdf)

    existing_pca512_features = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/feature_pca_jobs/ApplyPcaToFeatureHDFJob.JfBOA9zHcXBA/"
        "output/features_layer15_pca512.hdf"
    )
    existing_peak_starts = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/segmenter_jobs/DetectPeaksFromScoreHDFJob.Bf3WdJsaZWqv/"
        "output/peak_segment_starts.hdf"
    )

    segment_representation_hdfs = create_partitioned_segment_representation_jobs(
        output_prefix=prefix_name + "/segment_representations/layer15_pca512_peak_prom0p02_full_train",
        start_frame_hdf=existing_peak_starts,
        feature_hdf=existing_pca512_features,
        num_partitions=10,
        downsample_rate=2,
    )
    clustering_outputs = create_faiss_segment_clustering_pipeline(
        output_prefix=prefix_name + "/segment_clustering/layer15_pca512_peak_prom0p02_full_train_k60_4m",
        segment_hdfs=segment_representation_hdfs,
        num_clusters=60,
        num_samples=4_000_000,
        random_seed=1,
        use_cache_manager=False,
        train_gpu_mem=24,
        assign_gpu_mem=24,
    )
    clustering_outputs_k40 = create_faiss_segment_clustering_pipeline(
        output_prefix=prefix_name + "/segment_clustering/layer15_pca512_peak_prom0p02_full_train_k40_4m",
        segment_hdfs=segment_representation_hdfs,
        num_clusters=40,
        num_samples=4_000_000,
        random_seed=1,
        use_cache_manager=False,
        train_gpu_mem=24,
        assign_gpu_mem=24,
    )
    create_faiss_segment_clustering_pipeline(
        output_prefix=prefix_name + "/segment_clustering/layer15_pca512_peak_prom0p02_full_train_k128_4m",
        segment_hdfs=segment_representation_hdfs,
        num_clusters=128,
        num_samples=4_000_000,
        random_seed=1,
        use_cache_manager=False,
        train_gpu_mem=24,
        assign_gpu_mem=24,
    )
    create_faiss_segment_clustering_pipeline(
        output_prefix=prefix_name + "/segment_clustering/layer15_pca512_peak_prom0p02_full_train_k384_4m",
        segment_hdfs=segment_representation_hdfs,
        num_clusters=384,
        num_samples=4_000_000,
        random_seed=1,
        use_cache_manager=False,
        train_gpu_mem=24,
        assign_gpu_mem=24,
    )
    create_faiss_segment_clustering_pipeline(
        output_prefix=prefix_name + "/segment_clustering/layer15_pca512_peak_prom0p02_full_train_k512_4m",
        segment_hdfs=segment_representation_hdfs,
        num_clusters=512,
        num_samples=4_000_000,
        random_seed=1,
        use_cache_manager=False,
        train_gpu_mem=24,
        assign_gpu_mem=24,
    )

    alignment_hdfs = [tk.Path(f"output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf") for i in range(1, 201)]
    phoneme_vocab = tk.Path(
        "/work/asr4/zyang/corpora/librispeech/960/vocab/"
        "train-merged-960.phon-corpus.sil-augmented.no_hash.bert.vocab"
    )
    cluster_vocab = tk.Path(
        "/work/asr4/zyang/corpora/librispeech/960/vocab/"
        "train-merged-960.layer15_pca512_peak_prom0p02.k60.cluster_ids.merged.bert.vocab"
    )
    cluster_vocab_k40 = tk.Path(
        "/work/asr4/zyang/corpora/librispeech/960/vocab/"
        "train-merged-960.layer15_pca512_peak_prom0p02.k40.cluster_ids.merged.bert.vocab"
    )
    gmm_idx_to_phoneme = tk.Path(
        "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc/"
        "example_setups/librispeech/phmm/misc/dump_train_other_960_eow_alignment_hdf/"
        "GenerateEowMonophoneStateTyingJob.LcucMA7avmoo/output/idx_to_phoneme.txt"
    )
    utterance_select = tk.Path("/work/asr4/zyang/corpora/librispeech/960/mlm_splits/dev1.indices.txt")
    transport_plans = {
        "l2sq": tk.Path("/u/zyang/setups/mini/script/mlm_embedding_l2sq_ot_transport_plan.npz"),
        "one_minus_cosine": tk.Path("/u/zyang/setups/mini/script/mlm_embedding_one_minus_cosine_ot_transport_plan.npz"),
        "l2sq_linear_map_iter": tk.Path(
            "/u/zyang/setups/mini/script/mlm_embedding_l2sq_ot_linear_map_iter_final_transport.npz"
        ),
        "l2sq_emd_linear_map_iter": tk.Path(
            "/u/zyang/setups/mini/script/mlm_embedding_l2sq_emd_linear_map_iter_final_transport.npz"
        ),
    }
    for name, transport_plan in transport_plans.items():
        per_job = ComputeClusterToGmmPerJob(
            alignment_hdfs=alignment_hdfs,
            cluster_hdfs=clustering_outputs["assignments"],
            transport_npz=transport_plan,
            phoneme_vocab=phoneme_vocab,
            cluster_vocab=cluster_vocab,
            gmm_idx_to_phoneme=gmm_idx_to_phoneme,
            utterance_select=utterance_select,
            output_filename=f"cluster_to_gmm_per_{name}.txt",
            mem_rqmt=24,
            time_rqmt=5,
        )
        per_job.add_alias(prefix_name + f"/cluster_to_gmm_per/{name}")
        tk.register_output(prefix_name + f"/cluster_to_gmm_per/{name}.txt", per_job.out_report)

    transport_plans_k40 = {
        "k40_l2sq_sinkhorn_eps0p1": tk.Path(
            "/u/zyang/setups/mini/script/mlm_embedding_k40_l2sq_sinkhorn_eps0p1_transport.npz"
        ),
        "k40_l2sq_sinkhorn_eps0p01": tk.Path(
            "/u/zyang/setups/mini/script/mlm_embedding_k40_l2sq_sinkhorn_eps0p01_transport.npz"
        ),
    }
    for name, transport_plan in transport_plans_k40.items():
        per_job = ComputeClusterToGmmPerJob(
            alignment_hdfs=alignment_hdfs,
            cluster_hdfs=clustering_outputs_k40["assignments"],
            transport_npz=transport_plan,
            phoneme_vocab=phoneme_vocab,
            cluster_vocab=cluster_vocab_k40,
            gmm_idx_to_phoneme=gmm_idx_to_phoneme,
            utterance_select=utterance_select,
            output_filename=f"cluster_to_gmm_per_{name}.txt",
            mem_rqmt=24,
            time_rqmt=5,
        )
        per_job.add_alias(prefix_name + f"/cluster_to_gmm_per/{name}")
        tk.register_output(prefix_name + f"/cluster_to_gmm_per/{name}.txt", per_job.out_report)


py = eow_phon_phmm_ls960_wav2vec2_large_layer15_pca
