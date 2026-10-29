"""
Cluster Adapter for Cross-Modal and Cheating Segmentation Setups.

Supports switching data and model paths dynamically based on the cluster setting:
- 'curium': Local I6 cluster (e.g., /work/smt4/..., /work/asr4/..., /u/zyang/...)
- 'rwth-hpc' (or 'rwth-hcp'): RWTH HPC Cluster (e.g., /rwthfs/rz/cluster/hpcwork/..., /rwthfs/rz/cluster/home/...)
"""

import os
import glob
from typing import List, Optional, Union, Dict
from sisyphus import tk


# Global active cluster setting. Default is "curium" when running on curium.
_ACTIVE_CLUSTER = "curium"


def set_cluster(cluster_name: str):
    """
    Sets the active cluster environment.
    Supported values: 'curium', 'rwth-hpc', 'rwth-hcp'.
    """
    global _ACTIVE_CLUSTER
    normalized = cluster_name.lower().replace("_", "-")
    if normalized in ("rwth-hcp", "rwth-hpc", "hpc", "rwth"):
        _ACTIVE_CLUSTER = "rwth-hpc"
    elif normalized in ("curium", "local"):
        _ACTIVE_CLUSTER = "curium"
    else:
        _ACTIVE_CLUSTER = cluster_name


def get_cluster() -> str:
    """Returns the currently active cluster name."""
    return _ACTIVE_CLUSTER


def is_curium() -> bool:
    """Checks if running under curium cluster configuration."""
    return _ACTIVE_CLUSTER == "curium"


class ClusterPaths:
    """Central path resolver for both curium and rwth-hpc environments."""

    # ---------------------------------------------------------
    # Curium Cluster Paths (Local)
    # ---------------------------------------------------------
    CURIUM_ZYANG_ORACLE_DIR = "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/ls960_gmm_oracle_segment_clustering"
    CURIUM_LOCAL_ROOT = "/u/benjamin.oyarzun/experiments/2026_05_07_first_experiments"
    CURIUM_ALIAS_ROOT = "/u/benjamin.oyarzun/experiments/2026_05_07_first_experiments/alias/experiments"
    CURIUM_WORK_ROOT = "/work/smt4/benjamin.oyarzun/work"

    # ---------------------------------------------------------
    # RWTH-HPC Cluster Paths
    # ---------------------------------------------------------
    HPC_WORK_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090"
    HPC_CHEATING_SEG_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/cheating_segmentation"
    HPC_HUBERT_DATA_ROOT = "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/hubert_cluster_indices"
    HPC_EMBEDDINGS_ROOT = "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/cross_modal_embeddings"
    HPC_ALIAS_ROOT = "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/alias/experiments/self_learning_emb_asr"


def get_cheating_segmentation_path(k_clusters: int = 512, cluster: Optional[str] = None) -> tk.Path:
    """
    Returns the path to the oracle (cheating) segmentation files / directory.
    - If cluster is 'curium': Uses /u/zyang/setups/mini/.../clustering/k{k_clusters}/compat
    - If cluster is 'rwth-hpc': Uses $HPCWORK/cheating_segmentation/ls960_gmm_oracle_segment_clustering/...
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    if norm in ("curium", "local"):
        candidate_compat = os.path.join(
            ClusterPaths.CURIUM_ZYANG_ORACLE_DIR,
            f"clustering/k{k_clusters}/compat",
        )
        if os.path.exists(candidate_compat):
            return tk.Path(candidate_compat)

        target_k = 128 if k_clusters <= 128 else (256 if k_clusters <= 300 else 512)
        fallback_compat = os.path.join(
            ClusterPaths.CURIUM_ZYANG_ORACLE_DIR,
            f"clustering/k{target_k}/compat",
        )
        if os.path.exists(fallback_compat):
            return tk.Path(fallback_compat)

        candidate_k = os.path.join(
            ClusterPaths.CURIUM_ZYANG_ORACLE_DIR,
            f"clustering/k{k_clusters}",
        )
        if os.path.exists(candidate_k):
            return tk.Path(candidate_k)

        return tk.Path(ClusterPaths.CURIUM_ZYANG_ORACLE_DIR)

    # RWTH-HPC / Default
    specific_path = os.path.join(
        ClusterPaths.HPC_CHEATING_SEG_DIR,
        f"ls960_gmm_oracle_segment_clustering/clustering/k{k_clusters}/compat",
    )
    if os.path.exists(specific_path):
        return tk.Path(specific_path)

    target_k = 128 if k_clusters <= 128 else (256 if k_clusters <= 300 else 512)
    fallback_hpc = os.path.join(
        ClusterPaths.HPC_CHEATING_SEG_DIR,
        f"ls960_gmm_oracle_segment_clustering/clustering/k{target_k}/compat",
    )
    if os.path.exists(fallback_hpc):
        return tk.Path(fallback_hpc)

    for root in [ClusterPaths.HPC_CHEATING_SEG_DIR, ClusterPaths.HPC_WORK_DIR]:
        candidate = os.path.join(root, f"k{k_clusters}")
        if os.path.exists(candidate):
            return tk.Path(candidate)

    return tk.Path(ClusterPaths.HPC_CHEATING_SEG_DIR)


def get_cheating_train_clusters_hdfs(num_clusters: int = 512, cluster: Optional[str] = None) -> List[tk.Path]:
    """
    Returns list of 20 sanitized and Returnn-compatible HDF files for oracle cheating clusters.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    if norm in ("curium", "local"):
        base_dir = os.path.join(
            ClusterPaths.CURIUM_ZYANG_ORACLE_DIR,
            f"clustering/k{num_clusters}/compat",
        )
        return [
            tk.Path(os.path.join(base_dir, f"cluster_labels_k{num_clusters}.{idx:03d}.returnn_compat.hdf"))
            for idx in range(20)
        ]

    # RWTH-HPC
    base_dir = os.path.join(
        ClusterPaths.HPC_CHEATING_SEG_DIR,
        f"ls960_gmm_oracle_segment_clustering/clustering/k{num_clusters}/compat",
    )
    return [
        tk.Path(os.path.join(base_dir, f"cluster_labels_k{num_clusters}.{idx:03d}.returnn_compat.hdf"))
        for idx in range(20)
    ]


def get_cheating_phoneme_ref_hdfs(cluster: Optional[str] = None) -> List[tk.Path]:
    """
    Returns list of 20 GMM phoneme reference HDF files (20 ms frame rate, silence=0)
    used to define oracle segment boundaries across LibriSpeech 960 (including train-clean-100).
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    if norm in ("curium", "local"):
        base_dir = (
            "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
            "ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame/phoneme_ref_from_gmm_vad"
        )
        return [
            tk.Path(os.path.join(base_dir, f"phoneme_ref.{idx:03d}.hdf"))
            for idx in range(20)
        ]

    # RWTH-HPC fallback
    base_dir = os.path.join(
        ClusterPaths.HPC_CHEATING_SEG_DIR,
        "ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame/phoneme_ref_from_gmm_vad",
    )
    return [
        tk.Path(os.path.join(base_dir, f"phoneme_ref.{idx:03d}.hdf"))
        for idx in range(20)
    ]


def get_hubert_cluster_hdfs(
    cluster_mult: Union[int, str],
    k_clusters: int,
    split: str = "train_other_960",
    cluster: Optional[str] = None,
) -> List[str]:
    """
    Finds precomputed HuBERT cluster HDF files across cluster environments.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    clus_dir_map = {
        1: "clus_1P_41",
        2: "clus_2P_82",
        4: "clus_4P_164",
        200: "clus_200_200",
        40: "clus_1P_40",
        80: "clus_2P_80",
        160: "clus_4P_160",
        250: "clus_250_250",
        500: "clus_500_500",
        483: "clus_483_483",
        6161: "clus_6161_6161",
    }
    if isinstance(cluster_mult, str) and cluster_mult.startswith("clus_"):
        c_name = cluster_mult
    else:
        c_name = clus_dir_map.get(cluster_mult, f"clus_{cluster_mult}P_{k_clusters}" if isinstance(cluster_mult, int) else f"clus_{cluster_mult}")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/data"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            ClusterPaths.HPC_WORK_DIR,
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            ClusterPaths.HPC_HUBERT_DATA_ROOT,
            os.path.dirname(ClusterPaths.HPC_HUBERT_DATA_ROOT),
        ])

    for root_dir in search_roots:
        target_dir = os.path.join(root_dir, c_name)
        if os.path.exists(target_dir):
            if split == "dev_other":
                files = glob.glob(os.path.join(target_dir, "*dev_other*.hdf"))
            else:
                files = (
                    glob.glob(os.path.join(target_dir, "*train_other*.hdf"))
                    or glob.glob(os.path.join(target_dir, "*train*.hdf"))
                    or glob.glob(os.path.join(target_dir, "*.hdf"))
                )
            if files:
                return sorted(files)

            # Fallback to raw_clusters.txt or collapsed_tokens.txt if no HDF exists
            for fname in ["raw_clusters.txt", "collapsed_tokens.txt"]:
                cand = os.path.join(target_dir, fname)
                if os.path.exists(cand):
                    return [cand]

    return []


def resolve_mapped_audio_embeddings(candidate_path: str) -> str:
    """
    Ensures mapped_audio_embeddings.npy is returned if it exists alongside W_star.npy
    or in the symlink target directory.
    """
    if not candidate_path:
        return ""
    if os.path.basename(candidate_path) == "mapped_audio_embeddings.npy" and os.path.exists(candidate_path):
        return candidate_path

    # Check sibling in same dir
    sib = os.path.join(os.path.dirname(candidate_path), "mapped_audio_embeddings.npy")
    if os.path.exists(sib):
        return sib

    # Check realpath dir (target of symlink)
    real_dir = os.path.dirname(os.path.realpath(candidate_path))
    real_mapped = os.path.join(real_dir, "mapped_audio_embeddings.npy")
    if os.path.exists(real_mapped):
        return real_mapped

    return candidate_path


# =============================================================================
# v1 Path Resolvers (Legacy 1P_41, 2P_82, 4P_164, 200_200)
# =============================================================================

def get_audio_embeddings_path(
    cluster_mult: Union[int, str] = 1,
    init_method: str = "freq",
    seed: int = 42,
    cluster: Optional[str] = None,
) -> str:
    """
    Finds mapped audio embeddings for v1 cluster ablation.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    cluster_dir_map = {
        1: "clus_1P_41",
        2: "clus_2P_82",
        4: "clus_4P_164",
        200: "clus_200_200",
        "200": "clus_200_200",
        "200_200": "clus_200_200",
    }
    c_name = cluster_dir_map.get(
        cluster_mult,
        f"clus_{cluster_mult}P_{cluster_mult * 41}" if isinstance(cluster_mult, int) else f"clus_{cluster_mult}"
    )
    align_dirs = ["align_freq", "align_init_freq"] if init_method == "freq" else [f"align_rand_seed{seed}", f"align_init_rand_seed{seed}"]

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"),
            ClusterPaths.HPC_ALIAS_ROOT,
            ClusterPaths.HPC_EMBEDDINGS_ROOT,
        ])

    for root in search_roots:
        for adir in align_dirs:
            for sub in ["", "output"]:
                target_dir = os.path.join(root, "train-clean-100", "hubert_l6", c_name, adir, sub) if sub else os.path.join(root, "train-clean-100", "hubert_l6", c_name, adir)
                target = os.path.join(target_dir, "mapped_audio_embeddings.npy")
                if os.path.exists(target):
                    return target
                target_wstar = os.path.join(target_dir, "W_star.npy")
                if os.path.exists(target_wstar):
                    return resolve_mapped_audio_embeddings(target_wstar)

    return ""


def get_text_embeddings_path(cluster: Optional[str] = None) -> str:
    """Finds text phoneme embeddings for v1."""
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"),
            ClusterPaths.HPC_ALIAS_ROOT,
            ClusterPaths.HPC_EMBEDDINGS_ROOT,
        ])

    for root in search_roots:
        for fname in ["phoneme_embeddings.npy", "embeddings.npy"]:
            candidate = os.path.join(root, fname)
            if os.path.exists(candidate):
                return candidate
            candidate_nested = os.path.join(root, "train-clean-100", "hubert_l6", "text_w2v_phonemes", "output", fname)
            if os.path.exists(candidate_nested):
                return candidate_nested

    return ""


def get_phoneme_vocab_path(cluster: Optional[str] = None) -> tk.Path:
    """Finds phoneme vocab.txt path for v1 / monophones."""
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/cross_modal_embeddings"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"),
            ClusterPaths.HPC_ALIAS_ROOT,
        ])

    for root in search_roots:
        for fname in ["phoneme_vocab.txt", "vocab.txt"]:
            target = os.path.join(root, fname)
            if os.path.exists(target):
                return tk.Path(target)
            target_nested = os.path.join(root, "train-clean-100", "hubert_l6", "text_w2v_phonemes", "output", fname)
            if os.path.exists(target_nested):
                return tk.Path(target_nested)

    if norm in ("curium", "local"):
        return tk.Path(os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text/phoneme_vocab.txt"))
    return tk.Path(os.path.join(ClusterPaths.HPC_WORK_DIR, "data/cross_modal_embeddings/vocab.txt"))


# =============================================================================
# v2 Path Resolvers (Monophones: 1P_40, 2P_80, 4P_160, 200_200)
# =============================================================================

V2_CLUSTER_CONFIGS = [
    (1, 40, "clus_1P_40"),
    (2, 80, "clus_2P_80"),
    (4, 160, "clus_4P_160"),
    (200, 200, "clus_200_200"),
]


def get_v2_audio_embeddings_path(
    clus_str: str = "clus_1P_40",
    cluster: Optional[str] = None,
) -> str:
    """
    Finds mapped audio embeddings for v2 monophone cluster ablation.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    c_name = clus_str if clus_str.startswith("clus_") else f"clus_{clus_str}"

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
        ])

    for root in search_roots:
        align_freq_dir = os.path.join(root, c_name, "align_freq")
        target_mapped = os.path.join(align_freq_dir, "mapped_audio_embeddings.npy")
        if os.path.exists(target_mapped):
            return target_mapped
        target_wstar = os.path.join(align_freq_dir, "W_star.npy")
        if os.path.exists(target_wstar):
            return resolve_mapped_audio_embeddings(target_wstar)

    return ""


def get_v2_text_embeddings_path(cluster: Optional[str] = None) -> str:
    """Finds text phoneme embeddings for v2."""
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
        ])

    for root in search_roots:
        cand = os.path.join(root, "phoneme_embeddings.npy")
        if os.path.exists(cand):
            return cand

    return ""


def get_v2_phoneme_vocab_path(cluster: Optional[str] = None) -> tk.Path:
    """Finds text phoneme vocabulary for v2."""
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6/text"),
        ])

    for root in search_roots:
        cand = os.path.join(root, "phoneme_vocab.txt")
        if os.path.exists(cand):
            return tk.Path(cand)

    return get_phoneme_vocab_path(cluster=cluster)


# =============================================================================
# v3 Path Resolvers (Diphones & Triphones: K in {200, 250, 500}, win in {3, 5, 10})
# =============================================================================

V3_CODE_TYPES = ["diphone", "triphone"]
V3_CLUSTER_COUNTS = [
    (200, "clus_200_200"),
    (250, "clus_250_250"),
    (500, "clus_500_500"),
    (483, "clus_483_483"),
    (6161, "clus_6161_6161"),
]
V3_WINDOWS = [1, 2, 3, 5, 10]
V3_DIMS = [64, 128, 300]


def get_v3_audio_embeddings_path(
    code_type: str,
    clus_str: str,
    win: int = 3,
    dim: Optional[int] = None,
    retrieval_metric: str = "cosine",
    cluster: Optional[str] = None,
) -> str:
    """
    Finds mapped audio embeddings for v3 (diphone / triphone, cluster size, context window, dimension, metric).
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    c_name = clus_str if clus_str.startswith("clus_") else f"clus_{clus_str}"

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_w2v2_matched_cov90_cheat_seg_pca512/train-clean-100/w2v2_large60k_l15"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_wav2vec2_matched_cov90_cheat_seg_pca512/train-clean-100/w2v2_large60k_l15"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_w2v2_matched_cov90_cheat_seg_pca512/train-clean-100/w2v2_large60k_l15"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_wav2vec2_matched_cov90_cheat_seg_pca512/train-clean-100/w2v2_large60k_l15"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
        ])

    dw_candidates = []
    if dim is not None and dim != 300:
        dw_candidates.append(f"dim{dim}_win{win}")
    dw_candidates.append(f"win{win}")

    metric_subdirs = []
    if retrieval_metric != "cosine":
        metric_subdirs.append(f"align_freq_{retrieval_metric}")
    metric_subdirs.append("align_freq")

    for root in search_roots:
        for dw_tag in dw_candidates:
            for m_sub in metric_subdirs:
                align_freq_dir = os.path.join(root, code_type, c_name, dw_tag, m_sub)
                target_mapped = os.path.join(align_freq_dir, "mapped_audio_embeddings.npy")
                if os.path.exists(target_mapped):
                    return target_mapped
                target_wstar = os.path.join(align_freq_dir, "W_star.npy")
                if os.path.exists(target_wstar):
                    return resolve_mapped_audio_embeddings(target_wstar)

    return ""


def get_v3_text_embeddings_path(
    code_type: str,
    win: int = 3,
    dim: Optional[int] = None,
    cluster: Optional[str] = None,
) -> str:
    """Finds text embeddings for v3/v5 (code_type, win, dim)."""
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5.3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6/text"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5.3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6/text"),
        ])

    dw_candidates = []
    if dim is not None and dim != 300:
        dw_candidates.append(f"dim{dim}_win{win}")
    dw_candidates.append(f"win{win}")

    for root in search_roots:
        for dw_tag in dw_candidates:
            cand = os.path.join(root, code_type, dw_tag, "embeddings.npy")
            if os.path.exists(cand):
                return cand

    return ""


def get_v3_vocab_path(
    code_type: str,
    win: int = 3,
    dim: Optional[int] = None,
    cluster: Optional[str] = None,
) -> tk.Path:
    """Finds text vocabulary for v3/v5 (code_type, win, dim)."""
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5.3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6/text"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5.3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6/text"),
        ])

    dw_candidates = []
    if dim is not None and dim != 300:
        dw_candidates.append(f"dim{dim}_win{win}")
    dw_candidates.append(f"win{win}")

    for root in search_roots:
        for dw_tag in dw_candidates:
            cand = os.path.join(root, code_type, dw_tag, "vocab.txt")
            if os.path.exists(cand):
                return tk.Path(cand)

    # Fallback to general vocabulary
    return get_phoneme_vocab_path(cluster=active)


def get_cluster_counts_path(
    clus_str: str = "clus_200_200",
    cluster: Optional[str] = None,
) -> Optional[tk.Path]:
    """
    Finds cluster_counts.txt for a given cluster setting.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")
    c_names = [clus_str]
    if not clus_str.startswith("clus_"):
        c_names.append(f"clus_{clus_str}")
    if "_" not in clus_str and clus_str.isdigit():
        c_names.append(f"clus_{clus_str}_{clus_str}")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"),
        ])

    for root in search_roots:
        for c_name in c_names:
            cand = os.path.join(root, c_name, "cluster_counts.txt")
            if os.path.exists(cand):
                return tk.Path(cand)

    return None


def get_raw_clusters_path(
    clus_str: str = "clus_200_200",
    cluster: Optional[str] = None,
) -> Optional[tk.Path]:
    """
    Finds raw_clusters.txt or collapsed_tokens.txt for a given cluster setting.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")
    c_names = [clus_str]
    if not clus_str.startswith("clus_"):
        c_names.append(f"clus_{clus_str}")
    if "_" not in clus_str and clus_str.isdigit():
        c_names.append(f"clus_{clus_str}_{clus_str}")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"),
        ])

    for root in search_roots:
        for c_name in c_names:
            for fname in ["raw_clusters.txt", "collapsed_tokens.txt"]:
                cand = os.path.join(root, c_name, fname)
                if os.path.exists(cand):
                    return tk.Path(cand)

    return None


def get_collapsed_clusters_path(
    clus_str: str = "clus_200_200",
    cluster: Optional[str] = None,
) -> Optional[tk.Path]:
    """
    Finds collapsed_tokens.txt (or collapsed_clusters.txt) for a given cluster setting.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")
    c_names = [clus_str]
    if not clus_str.startswith("clus_"):
        c_names.append(f"clus_{clus_str}")
    if "_" not in clus_str and clus_str.isdigit():
        c_names.append(f"clus_{clus_str}_{clus_str}")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca512/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v2/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"),
        ])

    for root in search_roots:
        for c_name in c_names:
            for fname in ["collapsed_tokens.txt", "collapsed_clusters.txt"]:
                cand = os.path.join(root, c_name, fname)
                if os.path.exists(cand):
                    return tk.Path(cand)

    return None


def get_hubert_seq_tags_path(dataset: str = "train-clean-100", cluster: Optional[str] = None) -> tk.Path:
    """
    Returns path to seq_tags.txt for HuBERT feature sequences.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")
    if norm in ("curium", "local"):
        p = os.path.join(
            ClusterPaths.CURIUM_LOCAL_ROOT,
            "work/i6_experiments/users/oyarzun/experiments/exp2026_09_15_self_learning_emb_asr/sis_recipe/data/hubert/ExtractHuBERTLayerFeaturesJob.kYveUlRLjWY1/output/seq_tags.txt"
        )
        if os.path.exists(p):
            return tk.Path(p)
    return tk.Path("/work/smt4/benjamin.oyarzun/work/i6_experiments/users/oyarzun/experiments/exp2026_09_15_self_learning_emb_asr/sis_recipe/data/hubert/ExtractHuBERTLayerFeaturesJob.kYveUlRLjWY1/output/seq_tags.txt")


def get_oracle_idx_to_phoneme_path(cluster: Optional[str] = None) -> tk.Path:
    """
    Returns path to idx_to_phoneme.txt mapping used by GMM oracle phoneme references.
    """
    return tk.Path("/u/zyang/setups/mini/output/lbs_mono_phone_eow/idx_to_phoneme.txt")


def get_v3_counts_path(
    code_type: str = "diphone",
    cluster: Optional[str] = None,
) -> Optional[tk.Path]:
    """
    Finds diphone or triphone counts.txt for v3/v5.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v5_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6/text"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v5_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90/train-clean-100/hubert_l6/text"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3/train-clean-100/hubert_l6/text"),
        ])

    for root in search_roots:
        cand = os.path.join(root, code_type, "counts.txt")
        if os.path.exists(cand):
            return tk.Path(cand)

    return None


def get_v4_cheat_seg_tokens_path(
    clus_str: str = "clus_483_483",
    cluster: Optional[str] = None,
) -> tk.Path:
    """
    Returns path to precomputed segment-clustered tokens (collapsed_tokens.txt)
    for v4 cheating segmentation K-Means.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, "output/experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, "experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg/train-clean-100/hubert_l6"),
        ])

    for root in search_roots:
        cand = os.path.join(root, clus_str, "collapsed_tokens.txt")
        if os.path.exists(cand):
            return tk.Path(cand)

    return tk.Path(f"output/experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg/train-clean-100/hubert_l6/{clus_str}/collapsed_tokens.txt")


def get_v5_cheat_seg_tokens_path(
    clus_str: str = "clus_483_483",
    cluster: Optional[str] = None,
    pca_dim: int = 512,
) -> Optional[tk.Path]:
    """
    Returns path to precomputed segment-clustered tokens (collapsed_tokens.txt)
    for v5/v5.3 cheating segmentation 512-dim PCA + K-Means, or None if not yet computed.
    Supports rectangular cluster names (e.g., clus_500_483, clus_500_6161) by also checking clus_500_500.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v3_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
        ])

    c_candidates = [clus_str]
    if clus_str.startswith("clus_500_") and "clus_500_500" not in c_candidates:
        c_candidates.append("clus_500_500")

    for root in search_roots:
        for c_name in c_candidates:
            cand = os.path.join(root, c_name, "collapsed_tokens.txt")
            if os.path.exists(cand):
                return tk.Path(cand)

    return None


def get_v5_3_gan_audio_embeddings_path(
    code_type: str,
    clus_str: str,
    win: int = 2,
    dim: int = 128,
    retrieval_metric: str = "csls",
    pca_dim: int = 512,
    cluster: Optional[str] = None,
) -> str:
    """
    Finds mapped audio embeddings for v5.3 GAN alignment (Chung et al., 2018 + CSLS Procrustes refinement).
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")
    c_name = clus_str if clus_str.startswith("clus_") else f"clus_{clus_str}"

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5.3_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6"),
        ])

    dw_candidates = [f"dim{dim}_win{win}", f"win{win}"]
    metric_subdirs = [f"align_gan_{retrieval_metric}", "align_gan"]

    for root in search_roots:
        for dw_tag in dw_candidates:
            for m_sub in metric_subdirs:
                align_gan_dir = os.path.join(root, code_type, c_name, dw_tag, m_sub)
                target_mapped = os.path.join(align_gan_dir, "mapped_audio_embeddings.npy")
                if os.path.exists(target_mapped):
                    return target_mapped
                target_wstar = os.path.join(align_gan_dir, "W_star.npy")
                if os.path.exists(target_wstar):
                    return resolve_mapped_audio_embeddings(target_wstar)

    return ""


def get_v5_4_cheat_seg_tokens_path(
    clus_str: str = "clus_483_483",
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    norm_tag: str = "unit_sphere",
) -> Optional[tk.Path]:
    """
    Returns path to precomputed segment-clustered tokens (collapsed_tokens.txt)
    for v5.4 cheating segmentation (unit-sphere or z-score normalized) 512-dim PCA + K-Means, or None if not yet computed.
    Supports rectangular cluster names (e.g., clus_500_483, clus_500_6161) by also checking clus_500_500.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    norm_clean = norm_tag.strip().lower()
    if norm in ("curium", "local"):
        search_roots.append(
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5.4_{norm_clean}_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )
        search_roots.append(
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5.4_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )
    else:
        search_roots.append(
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5.4_{norm_clean}_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )
        search_roots.append(
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5.4_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )

    c_candidates = [clus_str]
    if clus_str.startswith("clus_500_") and "clus_500_500" not in c_candidates:
        c_candidates.append("clus_500_500")

    for root in search_roots:
        for c_name in c_candidates:
            cand = os.path.join(root, c_name, "collapsed_tokens.txt")
            if os.path.exists(cand):
                return tk.Path(cand)

    return None


def get_v5_4_gan_audio_embeddings_path(
    code_type: str,
    clus_str: str,
    win: int = 2,
    dim: int = 128,
    retrieval_metric: str = "csls",
    pca_dim: int = 512,
    cluster: Optional[str] = None,
    norm_tag: str = "unit_sphere",
) -> str:
    """
    Finds mapped audio embeddings for v5.4 GAN alignment (Unit sphere or Z-score normalized + CSLS Procrustes refinement).
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")
    c_name = clus_str if clus_str.startswith("clus_") else f"clus_{clus_str}"
    norm_clean = norm_tag.strip().lower()

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.append(
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5.4_{norm_clean}_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )
        search_roots.append(
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5.4_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )
    else:
        search_roots.append(
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5.4_{norm_clean}_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )
        search_roots.append(
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5.4_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/hubert_l6")
        )

    dw_candidates = [f"dim{dim}_win{win}", f"win{win}"]
    metric_subdirs = [f"align_gan_{retrieval_metric}", "align_gan"]

    for root in search_roots:
        for dw_tag in dw_candidates:
            for m_sub in metric_subdirs:
                align_gan_dir = os.path.join(root, code_type, c_name, dw_tag, m_sub)
                target_mapped = os.path.join(align_gan_dir, "mapped_audio_embeddings.npy")
                if os.path.exists(target_mapped):
                    return target_mapped
                target_wstar = os.path.join(align_gan_dir, "W_star.npy")
                if os.path.exists(target_wstar):
                    return resolve_mapped_audio_embeddings(target_wstar)

    return ""


def get_v5_w2v2_cheat_seg_tokens_path(
    clus_str: str = "clus_483_483",
    cluster: Optional[str] = None,
    pca_dim: int = 512,
) -> Optional[tk.Path]:
    """
    Returns path to precomputed segment-clustered tokens (collapsed_tokens.txt)
    for v5-wav2vec2.0 cheating segmentation 512-dim PCA + K-Means, or None if not yet computed.
    """
    active = cluster if cluster is not None else get_cluster()
    norm = active.lower().replace("_", "-")

    search_roots = []
    if norm in ("curium", "local"):
        search_roots.extend([
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5_w2v2_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/w2v2_large60k_l15"),
            os.path.join(ClusterPaths.CURIUM_LOCAL_ROOT, f"output/experiments/self_learning_emb_asr_v5_wav2vec2_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/w2v2_large60k_l15"),
        ])
    else:
        search_roots.extend([
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5_w2v2_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/w2v2_large60k_l15"),
            os.path.join(ClusterPaths.HPC_WORK_DIR, f"experiments/self_learning_emb_asr_v5_wav2vec2_matched_cov90_cheat_seg_pca{pca_dim}/train-clean-100/w2v2_large60k_l15"),
        ])

    for root in search_roots:
        cand = os.path.join(root, clus_str, "collapsed_tokens.txt")
        if os.path.exists(cand):
            return tk.Path(cand)

    return None


def get_phoneme_text_corpus_path(cluster: Optional[str] = None) -> Optional[tk.Path]:
    """
    Returns the path to the LibriSpeech training phonemes corpus for N-gram Language Modeling.
    """
    active = get_cluster() if cluster is None else cluster
    paths = [
        os.path.join(
            ClusterPaths.CURIUM_LOCAL_ROOT,
            "work/i6_experiments/users/oyarzun/experiments/exp2026_09_15_self_learning_emb_asr/sis_recipe/data/text/PhonemizeTextDataJob.h2XaligKUcKZ/output/text.phonemes.txt",
        ),
        os.path.join(
            ClusterPaths.HPC_WORK_DIR,
            "sisyphus-work-dirs/exp2026_09_15_self_learning_emb_asr/data/text/PhonemizeTextDataJob.h2XaligKUcKZ/output/text.phonemes.txt",
        ),
    ]
    for p in paths:
        if os.path.exists(p):
            return tk.Path(p)
    return None





