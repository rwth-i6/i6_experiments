"""
Cluster Adapter for Unsupervised Neural Machine Translation (UNMT) Setups.

Supports switching data, tool, and model paths dynamically based on the cluster setting:
- 'curium': Local i6 cluster (/work/smt4/benjamin.oyarzun/..., /u/zyang/..., /u/corpora/...)
- 'rwth-hpc' (or 'rwth-hcp'): RWTH HPC Cluster (/rwthfs/rz/cluster/hpcwork/..., /rwthfs/rz/cluster/home/...)
"""

import os
import sys
from typing import List, Optional
from sisyphus import tk


def _detect_default_cluster() -> str:
    if os.path.exists("/work/smt4/benjamin.oyarzun") or os.path.exists("/u/corpora"):
        return "curium"
    return "rwth-hpc"


_ACTIVE_CLUSTER = _detect_default_cluster()


def set_cluster(cluster_name: str) -> None:
    """
    Sets the active cluster environment.
    Supported values: 'curium', 'rwth-hpc', 'rwth-hcp'.
    """
    global _ACTIVE_CLUSTER
    normalized = cluster_name.lower().replace("_", "-")
    if normalized in ("rwth-hcp", "rwth-hpc", "hpc", "rwth"):
        _ACTIVE_CLUSTER = "rwth-hpc"
    elif normalized in ("curium", "local", "i6"):
        _ACTIVE_CLUSTER = "curium"
    else:
        _ACTIVE_CLUSTER = cluster_name


def get_cluster() -> str:
    """Returns the currently active cluster name."""
    return _ACTIVE_CLUSTER


def is_curium(cluster: Optional[str] = None) -> bool:
    """Checks if running under curium (local i6) cluster configuration."""
    active = cluster if cluster is not None else get_cluster()
    return active.lower().replace("_", "-") in ("curium", "local", "i6")


class UNMTClusterPaths:
    """Central path resolver for both curium (i6) and rwth-hpc environments."""

    # ---------------------------------------------------------
    # Curium (Local i6) Cluster Paths
    # ---------------------------------------------------------
    CURIUM_SMT4_USER_DIR = "/work/smt4/benjamin.oyarzun"
    CURIUM_WMT14_DIR = "/work/smt4/benjamin.oyarzun/wmt14_de_en"
    CURIUM_COVOST2_TSV = "/work/smt4/benjamin.oyarzun/covost2_en_de/covost_v2.en_de.tsv"
    CURIUM_CONTAINER_LIBS = "/work/smt4/benjamin.oyarzun/container_libs"
    CURIUM_NLTK_DATA = "/work/smt4/benjamin.oyarzun/container_libs/nltk_data"
    CURIUM_VENV_SITE = "/work/smt4/benjamin.oyarzun/venv/lib/python3.10/site-packages"

    CURIUM_PHONEME_REF_DIR = (
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
        "ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame/phoneme_ref_from_gmm_vad"
    )
    CURIUM_DEV_OTHER_ALIGN_DIR = "/u/zyang/setups/mini/output/lbs_mono_phone_eow_lexicon_dev_other"
    CURIUM_LIBRISPEECH_DIR = "/u/corpora/speech/LibriSpeech/LibriSpeech"
    CURIUM_COMMONVOICE_EN_AUDIO_DIR = "/u/corpora/speech/CommonVoice/en-corpus-6.1-2020-12-11/audio"
    CURIUM_MUSTC_EN_DE_DIR = "/u/corpora/speechtranslation/must-c/en-de"

    # 20ms (50 Hz) wav2vec2 phoneme CTC model trained on LibriSpeech 960h (79 EOW phones + blank=79)
    CURIUM_WAV2VEC2_PHONEME_CTC_CKPT = (
        "/work/asr4/zyang/mini/work/i6_core/returnn/training/"
        "ReturnnTrainingJob.U3AWP7d5sCOC/output/models/epoch.200.pt"
    )
    CURIUM_WAV2VEC2_PHONEME_CTC_PRIOR = (
        "/work/asr4/zyang/mini/work/i6_core/returnn/forward/"
        "ReturnnForwardJobV2.RDZQ1sO4k3ui/output/prior_9.txt"
    )
    CURIUM_HF_CACHE_DIR = "/work/asr4/zyang/hf_cache"
    CURIUM_ESPEAK_APPTAINER = "/work/smt4/zeineldeen/enrique.leon.lozano/containers/Fairseq/wav2vec_u_v4_kaldi.sif"

    # ---------------------------------------------------------
    # RWTH-HPC Cluster Paths
    # ---------------------------------------------------------
    HPC_WORK_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090"
    HPC_WMT14_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/wmt14_de_en"
    HPC_COVOST2_TSV = "/rwthfs/rz/cluster/hpcwork/bpd03090/covost2_en_de/covost_v2.en_de.tsv"
    HPC_CONTAINER_LIBS = "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs"
    HPC_NLTK_DATA = "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/container_libs/nltk_data"

    HPC_PHONEME_REF_DIR = (
        "/rwthfs/rz/cluster/hpcwork/bpd03090/cheating_segmentation/"
        "ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame/phoneme_ref_from_gmm_vad"
    )
    HPC_DEV_OTHER_ALIGN_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/cheating_segmentation/lbs_mono_phone_eow_lexicon_dev_other"
    HPC_LIBRISPEECH_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/LibriSpeech"
    HPC_COMMONVOICE_EN_AUDIO_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/CommonVoice/en/clips"
    HPC_MUSTC_EN_DE_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/must-c/en-de"
    HPC_WAV2VEC2_PHONEME_CTC_CKPT = "/rwthfs/rz/cluster/hpcwork/bpd03090/models/wav2vec2_phoneme_ctc_epoch200.pt"
    HPC_WAV2VEC2_PHONEME_CTC_PRIOR = "/rwthfs/rz/cluster/hpcwork/bpd03090/models/wav2vec2_phoneme_ctc_prior_9.txt"
    HPC_HF_CACHE_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/hf_cache"


def get_wmt14_dir(cluster: Optional[str] = None) -> str:
    """Returns the root directory containing WMT14 monolingual and raw dev/test data."""
    if is_curium(cluster):
        return UNMTClusterPaths.CURIUM_WMT14_DIR
    raw_hpcwork = os.environ.get("HPCWORK", UNMTClusterPaths.HPC_WORK_DIR)
    if raw_hpcwork.startswith("/hpcwork/"):
        raw_hpcwork = "/rwthfs/rz/cluster/hpcwork/" + raw_hpcwork[len("/hpcwork/") :]
    cand = os.path.join(raw_hpcwork, "wmt14_de_en")
    if os.path.exists(cand):
        return cand
    return UNMTClusterPaths.HPC_WMT14_DIR


def get_container_libs_paths() -> List[str]:
    """
    Returns ordered candidate site-packages directories containing user-installed Python libraries
    (sacremoses, nltk, g2p_en, phonemizer, subword-nmt, etc.) across both clusters.
    """
    return [
        UNMTClusterPaths.CURIUM_CONTAINER_LIBS,
        UNMTClusterPaths.CURIUM_VENV_SITE,
        UNMTClusterPaths.HPC_CONTAINER_LIBS,
    ]


def setup_extra_python_paths() -> None:
    """
    Appends cluster container library paths to sys.path (without shadowing container-native
    compiled packages such as torch, numpy, scipy, or h5py).
    """
    for p in get_container_libs_paths():
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)


def get_nltk_data_paths() -> List[str]:
    """Returns candidate nltk_data directories across both clusters."""
    return [
        UNMTClusterPaths.CURIUM_NLTK_DATA,
        UNMTClusterPaths.HPC_NLTK_DATA,
    ]


def get_en_blank_train_hdfs(cluster: Optional[str] = None) -> List[tk.Path]:
    """
    Returns the 20 GMM VAD-resampled 20ms frame-level phoneme reference HDFs on LibriSpeech 960h
    (40 classes: 0=[SILENCE]/blank, 1..39=ARPAbet AA..ZH).
    """
    base_dir = (
        UNMTClusterPaths.CURIUM_PHONEME_REF_DIR
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_PHONEME_REF_DIR
    )
    return [tk.Path(os.path.join(base_dir, f"phoneme_ref.{idx:03d}.hdf")) for idx in range(20)]


def get_en_blank_dev_other_hdfs(cluster: Optional[str] = None) -> List[tk.Path]:
    """
    Returns the 10 GMM EOW phoneme alignment HDFs on LibriSpeech dev-other (10ms frame rate, 79 classes).
    """
    base_dir = (
        UNMTClusterPaths.CURIUM_DEV_OTHER_ALIGN_DIR
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_DEV_OTHER_ALIGN_DIR
    )
    return [tk.Path(os.path.join(base_dir, f"alignment_{idx}.hdf")) for idx in range(1, 11)]


def get_librispeech_corpus_dir(cluster: Optional[str] = None) -> tk.Path:
    """Returns the LibriSpeech root directory containing dev-other, test-other, etc."""
    path = (
        UNMTClusterPaths.CURIUM_LIBRISPEECH_DIR
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_LIBRISPEECH_DIR
    )
    return tk.Path(path)


def get_covost2_en_de_tsv(cluster: Optional[str] = None) -> tk.Path:
    """Returns path to CoVoST-2 English-to-German TSV metadata file."""
    path = (
        UNMTClusterPaths.CURIUM_COVOST2_TSV
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_COVOST2_TSV
    )
    return tk.Path(path)


def get_commonvoice_en_audio_dir(cluster: Optional[str] = None) -> tk.Path:
    """Returns path to CommonVoice English audio clips directory."""
    path = (
        UNMTClusterPaths.CURIUM_COMMONVOICE_EN_AUDIO_DIR
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_COMMONVOICE_EN_AUDIO_DIR
    )
    return tk.Path(path)


def get_mustc_en_de_dir(cluster: Optional[str] = None) -> tk.Path:
    """Returns path to MuST-C English-to-German speech translation directory."""
    path = (
        UNMTClusterPaths.CURIUM_MUSTC_EN_DE_DIR
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_MUSTC_EN_DE_DIR
    )
    return tk.Path(path)


def get_wav2vec2_phoneme_ctc_checkpoint(cluster: Optional[str] = None) -> tk.Path:
    """Returns path to the 20ms wav2vec2 phoneme CTC checkpoint (79 EOW phones + blank=79)."""
    path = (
        UNMTClusterPaths.CURIUM_WAV2VEC2_PHONEME_CTC_CKPT
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_WAV2VEC2_PHONEME_CTC_CKPT
    )
    return tk.Path(path)


def get_wav2vec2_phoneme_ctc_prior(cluster: Optional[str] = None) -> tk.Path:
    """Returns path to the layer-9 log-prior file for the 20ms wav2vec2 phoneme CTC checkpoint."""
    path = (
        UNMTClusterPaths.CURIUM_WAV2VEC2_PHONEME_CTC_PRIOR
        if is_curium(cluster)
        else UNMTClusterPaths.HPC_WAV2VEC2_PHONEME_CTC_PRIOR
    )
    return tk.Path(path)


def get_hf_cache_dir(cluster: Optional[str] = None) -> str:
    """Returns HuggingFace cache directory containing facebook/wav2vec2-base."""
    if is_curium(cluster):
        return UNMTClusterPaths.CURIUM_HF_CACHE_DIR
    return UNMTClusterPaths.HPC_HF_CACHE_DIR
