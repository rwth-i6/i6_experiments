"""
Config: LibriSpeech 960 (Cheating Segmentation) - Consecutive Representation Tests
Tests the Consecutive Segment Discriminability Assumption (Tests 1, 2, and 3)
across HuBERT, Spin-HuBERT, and Wav2Vec 2.0 backbones.
"""

import os
from typing import Dict, List, Optional
from sisyphus import tk

from ....cluster_adapter import (
    get_cheating_phoneme_ref_hdfs,
    get_cluster,
    set_cluster,
)
from ....analysis.representation_tests import (
    ConsecutiveRepresentationTestJob,
    ClusterPhoneCompositionTestJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_09_15_self_learning_emb_asr.sis_recipe.data.hubert import (
    ExtractHuBERTLayerFeaturesJob,
    HuBERTFramePCAFitJob,
    DownloadSpinHuBERTCheckpointJob,
)


DEFAULT_TEST4_CLUSTER_COUNTS: List[int] = [40, 120, 512]


MODEL_CONFIGS = {
    "hubert": {
        "model_path": "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
        "layers": [12, 6],
        "spin_codebook": None,
    },
    "hubert_l6": {
        "model_path": "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
        "layers": [6],
        "spin_codebook": None,
    },
    "hubert_l12": {
        "model_path": "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
        "layers": [12],
        "spin_codebook": None,
    },
    "spinhubert": {
        "model_path": "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
        "layers": [12],
        "spin_codebook": 2048,
    },
    "wav2vec2.0": {
        "model_path": "/work/smt4/benjamin.oyarzun/models/wav2vec2-large-lv60",
        "layers": [15],
        "spin_codebook": None,
    },
}


def _register_test4_subexp_outputs(
    base_prefix: str,
    pooling_prefix: str,
    p_method: str,
    k_val: int,
    t4_job: ConsecutiveRepresentationTestJob,
    c_method: str = "kmeans",
):
    """Registers Test 4 outputs under both the pooling-specific prefix and the model-level test4 folder."""
    is_spectral = c_method in ("spectral", "spectral_clustering")
    spec_folder = "spectral/" if is_spectral else ""

    # 1. Under pooling-specific test4 folder: {pooling_prefix}/test4_cluster_phone_composition/
    p_t4 = f"{pooling_prefix}/test4_cluster_phone_composition/{spec_folder}".rstrip("/")
    if k_val == 40:
        tk.register_output(f"{p_t4}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
        tk.register_output(f"{p_t4}/cluster_phone_composition.png", t4_job.out_t4_plot)
        tk.register_output(f"{p_t4}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
        tk.register_output(f"{p_t4}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)

    tk.register_output(f"{p_t4}/k{k_val}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
    tk.register_output(f"{p_t4}/k{k_val}/cluster_phone_composition.png", t4_job.out_t4_plot)
    tk.register_output(f"{p_t4}/k{k_val}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
    tk.register_output(f"{p_t4}/k{k_val}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)

    tk.register_output(f"{p_t4}/cluster_phone_composition_k{k_val}.json", t4_job.out_t4_summary_json)
    tk.register_output(f"{p_t4}/cluster_phone_composition_k{k_val}.png", t4_job.out_t4_plot)
    tk.register_output(f"{p_t4}/cluster_phone_composition_word_boundaries_k{k_val}.json", t4_job.out_t4_wb_summary_json)
    tk.register_output(f"{p_t4}/cluster_phone_composition_word_boundaries_k{k_val}.png", t4_job.out_t4_wb_plot)

    # 2. Under model-level test4 folder: {base_prefix}/test4_cluster_phone_composition/
    m_t4 = f"{base_prefix}/test4_cluster_phone_composition/{spec_folder}".rstrip("/")
    tk.register_output(f"{m_t4}/k{k_val}_{p_method}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
    tk.register_output(f"{m_t4}/k{k_val}_{p_method}/cluster_phone_composition.png", t4_job.out_t4_plot)
    tk.register_output(f"{m_t4}/k{k_val}_{p_method}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
    tk.register_output(f"{m_t4}/k{k_val}_{p_method}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)

    tk.register_output(f"{m_t4}/{p_method}/k{k_val}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
    tk.register_output(f"{m_t4}/{p_method}/k{k_val}/cluster_phone_composition.png", t4_job.out_t4_plot)
    tk.register_output(f"{m_t4}/{p_method}/k{k_val}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
    tk.register_output(f"{m_t4}/{p_method}/k{k_val}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)

    if not is_spectral:
        if p_method == "mean":
            if k_val == 40:
                tk.register_output(f"{m_t4}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
                tk.register_output(f"{m_t4}/cluster_phone_composition.png", t4_job.out_t4_plot)
                tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
                tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition.png", t4_job.out_t4_plot)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)

            tk.register_output(f"{m_t4}/cluster_phone_composition_k{k_val}.json", t4_job.out_t4_summary_json)
            tk.register_output(f"{m_t4}/cluster_phone_composition_k{k_val}.png", t4_job.out_t4_plot)
            tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries_k{k_val}.json", t4_job.out_t4_wb_summary_json)
            tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries_k{k_val}.png", t4_job.out_t4_wb_plot)
        else:
            if k_val == 40:
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition.png", t4_job.out_t4_plot)
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)

            tk.register_output(f"{m_t4}/cluster_phone_composition_k{k_val}_{p_method}.json", t4_job.out_t4_summary_json)
            tk.register_output(f"{m_t4}/cluster_phone_composition_k{k_val}_{p_method}.png", t4_job.out_t4_plot)
            tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries_k{k_val}_{p_method}.json", t4_job.out_t4_wb_summary_json)
            tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries_k{k_val}_{p_method}.png", t4_job.out_t4_wb_plot)
    else:
        if p_method == "mean":
            if k_val == 40:
                tk.register_output(f"{m_t4}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
                tk.register_output(f"{m_t4}/cluster_phone_composition.png", t4_job.out_t4_plot)
                tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
                tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition.png", t4_job.out_t4_plot)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
            tk.register_output(f"{m_t4}/k{k_val}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)
        else:
            if k_val == 40:
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition.json", t4_job.out_t4_summary_json)
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition.png", t4_job.out_t4_plot)
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition_word_boundaries.json", t4_job.out_t4_wb_summary_json)
                tk.register_output(f"{m_t4}/{p_method}/cluster_phone_composition_word_boundaries.png", t4_job.out_t4_wb_plot)
            tk.register_output(f"{m_t4}/cluster_phone_composition_k{k_val}_{p_method}.json", t4_job.out_t4_summary_json)
            tk.register_output(f"{m_t4}/cluster_phone_composition_k{k_val}_{p_method}.png", t4_job.out_t4_plot)
            tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries_k{k_val}_{p_method}.json", t4_job.out_t4_wb_summary_json)
            tk.register_output(f"{m_t4}/cluster_phone_composition_word_boundaries_k{k_val}_{p_method}.png", t4_job.out_t4_wb_plot)


def build_representation_tests_pipeline(
    cluster: str = "curium",
    models: Optional[List[str]] = None,
    num_utts: int = 1000,
    pca_dim: int = 512,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
    clustering_methods: Optional[List[str]] = None,
    spectral_n_neighbors: int = 30,
    spectral_sample_size: int = 25000,
) -> Dict[str, ConsecutiveRepresentationTestJob]:
    """
    Builds the Sisyphus graph for representation tests 1 to 4 across the specified models,
    pooling methods (e.g. 'mean', 'center_weighted'), clustering methods ('kmeans', 'spectral'),
    and Test 4 cluster counts (default: K=40 [v5.9 baseline], K=120 [v5.10 allophonic scale], K=512 [Test 3 fine scale]).
    For HuBERT, creates separate subfolders for Layer 12 and Layer 6:
      output/unsup_audio_cluster_to_phoneme/librispeech/cross_modal_cheat_seg_{cluster}/representation_tests/hubert/layer_12/{pooling_method}/
      output/unsup_audio_cluster_to_phoneme/librispeech/cross_modal_cheat_seg_{cluster}/representation_tests/hubert/layer_6/{pooling_method}/
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()

    if models is None:
        models = ["hubert", "spinhubert", "wav2vec2.0"]

    if pooling_methods is None:
        pooling_methods = ["mean", "center_weighted"]

    if test4_cluster_counts is None:
        test4_cluster_counts = list(DEFAULT_TEST4_CLUSTER_COUNTS)

    if clustering_methods is None:
        clustering_methods = ["kmeans", "spectral"]

    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", "train-clean-100"))
    seg_hdfs = get_cheating_phoneme_ref_hdfs(cluster=active_cluster)

    feature_jobs_cache = {}
    pca_jobs_cache = {}
    test_jobs = {}

    for m in models:
        m_key = m.lower()
        cfg = MODEL_CONFIGS.get(m_key, MODEL_CONFIGS["hubert"])
        layers = cfg.get("layers", [cfg.get("layer", 12)])

        spin_ckpt = None
        if cfg.get("spin_codebook") is not None:
            dl_spin = DownloadSpinHuBERTCheckpointJob(codebook_size=int(cfg["spin_codebook"]))
            dl_spin.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_key}/download_spin_{cfg['spin_codebook']}")
            spin_ckpt = dl_spin.out_ckpt

        for layer in layers:
            feat_cache_key = (cfg["model_path"], layer, spin_ckpt)
            if feat_cache_key not in feature_jobs_cache:
                feat_job = ExtractHuBERTLayerFeaturesJob(
                    audio_dir=audio_dir,
                    model_name=cfg["model_path"],
                    layer=layer,
                    spin_ckpt=spin_ckpt,
                )
                if m_key.startswith("hubert"):
                    feat_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/hubert/layer_{layer}/features")
                else:
                    feat_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_key}/features")
                feature_jobs_cache[feat_cache_key] = feat_job
            else:
                feat_job = feature_jobs_cache[feat_cache_key]

            pca_comp = None
            pca_mean = None
            if pca_dim is not None:
                pca_cache_key = (feat_cache_key, pca_dim)
                if pca_cache_key not in pca_jobs_cache:
                    pca_job = HuBERTFramePCAFitJob(
                        features_npy=feat_job.out_features_npy,
                        n_components=pca_dim,
                    )
                    if m_key.startswith("hubert"):
                        pca_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/hubert/layer_{layer}/pca_{pca_dim}")
                    else:
                        pca_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_key}/pca_{pca_dim}")
                    pca_jobs_cache[pca_cache_key] = pca_job
                else:
                    pca_job = pca_jobs_cache[pca_cache_key]
                pca_comp = pca_job.out_components_npy
                pca_mean = pca_job.out_mean_npy

            for p_method in pooling_methods:
                model_display_name = f"hubert_l{layer}" if m_key.startswith("hubert") else m_key
                m_alias_part = f"hubert/layer_{layer}" if m_key.startswith("hubert") else m_key

                if m_key.startswith("hubert"):
                    l_prefix = f"unsup_audio_cluster_to_phoneme/librispeech/cross_modal_cheat_seg_{active_cluster}/representation_tests/hubert/layer_{layer}"
                else:
                    l_prefix = f"unsup_audio_cluster_to_phoneme/librispeech/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_key}"
                target_prefix = f"{l_prefix}/{p_method}"
                b_prefix = f"unsup_audio_cluster_to_phoneme/librispeech/cross_modal_cheat_seg_{active_cluster}/representation_tests/hubert"
                top_prefix = f"{b_prefix}/{p_method}"
                job_key = f"{m_key}_l{layer}_{p_method}" if m_key.startswith("hubert") else f"{m_key}_{p_method}"

                for c_method in clustering_methods:
                    is_spectral = c_method in ("spectral", "spectral_clustering")

                    test_job = ConsecutiveRepresentationTestJob(
                        features_npy=feat_job.out_features_npy,
                        lengths_txt=feat_job.out_lengths_txt,
                        seq_tags_txt=feat_job.out_seq_tags_txt,
                        segmentation_hdfs=seg_hdfs,
                        model_name=model_display_name,
                        pca_components_npy=pca_comp,
                        pca_mean_npy=pca_mean,
                        normalize_segments="unit_sphere",
                        num_utts=num_utts,
                        pooling_method=p_method,
                        clustering_method=c_method,
                        spectral_n_neighbors=spectral_n_neighbors,
                        spectral_sample_size=spectral_sample_size,
                    )

                    if not is_spectral:
                        test_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_alias_part}/{p_method}/test_suite")
                        tk.register_output(f"{target_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                        tk.register_output(f"{target_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                        tk.register_output(f"{target_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                        tk.register_output(f"{target_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                        tk.register_output(f"{target_prefix}/test3_kmeans_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                        tk.register_output(f"{target_prefix}/test3_kmeans_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)

                        if p_method == "mean":
                            if not m_key.startswith("hubert"):
                                test_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_key}/test_suite")
                            tk.register_output(f"{l_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                            tk.register_output(f"{l_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                            tk.register_output(f"{l_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                            tk.register_output(f"{l_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                            tk.register_output(f"{l_prefix}/test3_kmeans_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                            tk.register_output(f"{l_prefix}/test3_kmeans_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)

                        if m_key.startswith("hubert") and layer == 12:
                            tk.register_output(f"{top_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                            tk.register_output(f"{top_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                            tk.register_output(f"{top_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                            tk.register_output(f"{top_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                            tk.register_output(f"{top_prefix}/test3_kmeans_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                            tk.register_output(f"{top_prefix}/test3_kmeans_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)

                            if p_method == "mean":
                                tk.register_output(f"{b_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                                tk.register_output(f"{b_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                                tk.register_output(f"{b_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                                tk.register_output(f"{b_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                                tk.register_output(f"{b_prefix}/test3_kmeans_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                                tk.register_output(f"{b_prefix}/test3_kmeans_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)
                                test_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/hubert/test_suite")
                                test_jobs["hubert"] = test_job

                        test_jobs[job_key] = test_job
                    else:
                        spec_target_prefix = f"{target_prefix}/spectral"
                        spec_l_prefix = f"{l_prefix}/spectral"
                        spec_top_prefix = f"{top_prefix}/spectral"
                        spec_b_prefix = f"{b_prefix}/spectral"

                        test_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_alias_part}/{p_method}/spectral/test_suite")
                        tk.register_output(f"{spec_target_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                        tk.register_output(f"{spec_target_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                        tk.register_output(f"{spec_target_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                        tk.register_output(f"{spec_target_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                        tk.register_output(f"{spec_target_prefix}/test3_spectral_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                        tk.register_output(f"{spec_target_prefix}/test3_spectral_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)

                        if p_method == "mean":
                            if not m_key.startswith("hubert"):
                                test_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_key}/spectral/test_suite")
                            tk.register_output(f"{spec_l_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                            tk.register_output(f"{spec_l_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                            tk.register_output(f"{spec_l_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                            tk.register_output(f"{spec_l_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                            tk.register_output(f"{spec_l_prefix}/test3_spectral_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                            tk.register_output(f"{spec_l_prefix}/test3_spectral_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)

                        if m_key.startswith("hubert") and layer == 12:
                            tk.register_output(f"{spec_top_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                            tk.register_output(f"{spec_top_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                            tk.register_output(f"{spec_top_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                            tk.register_output(f"{spec_top_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                            tk.register_output(f"{spec_top_prefix}/test3_spectral_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                            tk.register_output(f"{spec_top_prefix}/test3_spectral_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)

                            if p_method == "mean":
                                tk.register_output(f"{spec_b_prefix}/test1_consecutive_distance/distance_summary.json", test_job.out_t1_summary_json)
                                tk.register_output(f"{spec_b_prefix}/test1_consecutive_distance/consecutive_vs_intra_distribution.png", test_job.out_t1_plot)
                                tk.register_output(f"{spec_b_prefix}/test2_supervised_discrimination/supervised_probe_metrics.json", test_job.out_t2_summary_json)
                                tk.register_output(f"{spec_b_prefix}/test2_supervised_discrimination/supervised_probe_summary.png", test_job.out_t2_plot)
                                tk.register_output(f"{spec_b_prefix}/test3_spectral_collision/kmeans_collision_summary.json", test_job.out_t3_summary_json)
                                tk.register_output(f"{spec_b_prefix}/test3_spectral_collision/collision_rate_vs_segment_duration.png", test_job.out_t3_plot)
                                test_job.add_alias(f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/hubert/spectral/test_suite")
                                test_jobs["hubert_spectral"] = test_job

                        test_jobs[f"{job_key}_spectral"] = test_job

                    # Test 4 Subexperiments across cluster counts (K=40, K=120 [v5.10], K=512 [Test 3])
                    for k_val in test4_cluster_counts:
                        k_int = int(k_val)
                        if k_int == 40:
                            t4_job = test_job
                        else:
                            t4_job = ClusterPhoneCompositionTestJob(
                                features_npy=feat_job.out_features_npy,
                                lengths_txt=feat_job.out_lengths_txt,
                                seq_tags_txt=feat_job.out_seq_tags_txt,
                                segmentation_hdfs=seg_hdfs,
                                model_name=model_display_name,
                                pca_components_npy=pca_comp,
                                pca_mean_npy=pca_mean,
                                normalize_segments="unit_sphere",
                                num_utts=num_utts,
                                pooling_method=p_method,
                                num_clusters=k_int,
                                clustering_method=c_method,
                                spectral_n_neighbors=spectral_n_neighbors,
                                spectral_sample_size=spectral_sample_size,
                            )
                            spec_alias_tag = "spectral/" if is_spectral else ""
                            t4_job.add_alias(
                                f"experiments/cross_modal_cheat_seg_{active_cluster}/representation_tests/{m_alias_part}/{p_method}/{spec_alias_tag}test4_k{k_int}".replace("//", "/")
                            )

                        _register_test4_subexp_outputs(
                            base_prefix=l_prefix,
                            pooling_prefix=target_prefix,
                            p_method=p_method,
                            k_val=k_int,
                            t4_job=t4_job,
                            c_method=c_method,
                        )
                        if m_key.startswith("hubert") and layer == 12:
                            _register_test4_subexp_outputs(
                                base_prefix=b_prefix,
                                pooling_prefix=top_prefix,
                                p_method=p_method,
                                k_val=k_int,
                                t4_job=t4_job,
                                c_method=c_method,
                            )

                        t4_key = f"{job_key}_spectral_test4_k{k_int}" if is_spectral else f"{job_key}_test4_k{k_int}"
                        test_jobs[t4_key] = t4_job

    return test_jobs


def run_representation_tests(
    cluster: str = "curium",
    models: Optional[List[str]] = None,
    num_utts: int = 1000,
    pca_dim: int = 512,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
    clustering_methods: Optional[List[str]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=models,
        num_utts=num_utts,
        pca_dim=pca_dim,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
        clustering_methods=clustering_methods,
    )


def representation_tests_hubert(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_hubert_l6(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert_l6"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_hubert_l12(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert_l12"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_spinhubert(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["spinhubert"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_spinhubert_center_weighted(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    """Executes the representation tests for Spin-HuBERT with center-weighted pooling."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["spinhubert"],
        num_utts=num_utts,
        pooling_methods=["center_weighted"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_spinhubert_mean(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    """Executes the representation tests for Spin-HuBERT with uniform mean pooling."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["spinhubert"],
        num_utts=num_utts,
        pooling_methods=["mean"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_hubert_l6_center_weighted(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert_l6"],
        num_utts=num_utts,
        pooling_methods=["center_weighted"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_hubert_l6_mean(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert_l6"],
        num_utts=num_utts,
        pooling_methods=["mean"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_hubert_l12_center_weighted(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert_l12"],
        num_utts=num_utts,
        pooling_methods=["center_weighted"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_hubert_l12_mean(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert_l12"],
        num_utts=num_utts,
        pooling_methods=["mean"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_wav2vec2(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["wav2vec2.0"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_wav2vec2_center_weighted(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["wav2vec2.0"],
        num_utts=num_utts,
        pooling_methods=["center_weighted"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_wav2vec2_mean(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["wav2vec2.0"],
        num_utts=num_utts,
        pooling_methods=["mean"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
    clustering_methods: Optional[List[str]] = None,
):
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["hubert", "spinhubert", "wav2vec2.0"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
        clustering_methods=clustering_methods,
    )


def representation_tests_mean(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return representation_tests(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=["mean"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_center_weighted(
    cluster: str = "curium",
    num_utts: int = 1000,
    test4_cluster_counts: Optional[List[int]] = None,
):
    return representation_tests(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=["center_weighted"],
        test4_cluster_counts=test4_cluster_counts,
    )


def representation_tests_test4(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    """Runs representation tests including Test 4 across K in {40, 120 (v5.10), 512 (Test 3)} and mean + center_weighted pooling."""
    return representation_tests(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts if test4_cluster_counts is not None else [40, 120, 512],
    )


def representation_tests_test4_k120(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
):
    """Runs Test 4 with K=120 clusters (v5.10 scale) across mean and center_weighted pooling."""
    return representation_tests(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=[120],
    )


def representation_tests_test4_k512(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
):
    """Runs Test 4 with K=512 clusters (Test 3 scale) across mean and center_weighted pooling."""
    return representation_tests(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=[512],
    )


representation_test = representation_tests
test4 = representation_tests_test4
test4_k120 = representation_tests_test4_k120
test4_k512 = representation_tests_test4_k512


def representation_tests_spectral(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    """Executes the representation tests strictly with Spectral Clustering."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
        clustering_methods=["spectral"],
    )


def representation_tests_spinhubert_spectral(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    """Executes the representation tests for Spin-HuBERT with Spectral Clustering."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        models=["spinhubert"],
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts,
        clustering_methods=["spectral"],
    )


def representation_tests_test4_spectral(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
    test4_cluster_counts: Optional[List[int]] = None,
):
    """Runs Test 4 specifically with Spectral Clustering across cluster counts and pooling methods."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=test4_cluster_counts if test4_cluster_counts is not None else [40, 120, 512],
        clustering_methods=["spectral"],
    )


def representation_tests_test4_spectral_k40(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
):
    """Runs Test 4 with K=40 clusters using Spectral Clustering."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=[40],
        clustering_methods=["spectral"],
    )


def representation_tests_test4_spectral_k120(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
):
    """Runs Test 4 with K=120 clusters using Spectral Clustering."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=[120],
        clustering_methods=["spectral"],
    )


def representation_tests_test4_spectral_k512(
    cluster: str = "curium",
    num_utts: int = 1000,
    pooling_methods: Optional[List[str]] = None,
):
    """Runs Test 4 with K=512 clusters using Spectral Clustering."""
    return build_representation_tests_pipeline(
        cluster=cluster,
        num_utts=num_utts,
        pooling_methods=pooling_methods,
        test4_cluster_counts=[512],
        clustering_methods=["spectral"],
    )


test4_spectral = representation_tests_test4_spectral
test4_spectral_k40 = representation_tests_test4_spectral_k40
test4_spectral_k120 = representation_tests_test4_spectral_k120
test4_spectral_k512 = representation_tests_test4_spectral_k512
spinhubert_spectral = representation_tests_spinhubert_spectral

