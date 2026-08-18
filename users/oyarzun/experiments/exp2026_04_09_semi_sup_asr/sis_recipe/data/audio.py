import os
from typing import Optional

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core.tools.download import DownloadJob
from i6_core.text.processing import PipelineJob, TailJob, HeadJob, ConcatenateJob

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import SetupFairseqJob
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.librispeech.data.audio_preprocessing import (
    Wav2VecUDeleteSilencesInAudioJob,
    Wav2VecUFeaturizeAudioJob,
)
from i6_experiments.users.schmitt.experiments.exp2025_10_02_shared_enc.librispeech.data.wav2vec import (
    DumpNumpyFeaturesToHdfJobV2,
    DumpClusterIndicesToHdfJob,
)

from ..default_tools import get_rvad_root


def remove_silences_from_audio(
    audio_dir: tk.Path,
    concurrent: int,
):
    environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")
    delete_silences_job = Wav2VecUDeleteSilencesInAudioJob(
        environment=environment,
        fairseq_root=SetupFairseqJob(
            tk.Path(
                "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
            ),
            environment,
        ).out_fairseq_root,
        audio_dir=audio_dir,
        valid_percent=0.01,
        extension="flac",
        rvad_root=get_rvad_root(),
        initial_manifests_dir=None,
        concurrent=concurrent,
        name_the_manifests_just_train_and_valid=True,
        max_n_audios_per_manifest=None,
    )

    return delete_silences_job.out_preprocessed_manifest


def featurize_audio(
    librispeech_key: str,
    input_audio_manifests: tk.Path,
    existing_clusters: Optional[tk.Path],
    existing_pca: Optional[tk.Path],
    featurize_concurrent: int,
    dump_hdf_concurrent: int,
    fixed_random_subset: Optional[int] = None,
    max_abs_value: Optional[float] = None,
    remove_cluster_repetitions: bool = False,
):
    w2v2_model_path = DownloadJob(
        "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt",
        target_filename="wav2vec_60kh_no_finetune.pt",
    ).out_file

    environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")
    featurize_job = Wav2VecUFeaturizeAudioJob(
        environment=tk.Path("/work/asr4/schmitt/venvs/fairseq_env"),
        fairseq_root=SetupFairseqJob(
            tk.Path(
                "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
            ),
            environment,
        ).out_fairseq_root,
        layer=14,
        existing_clusters=existing_clusters,
        existing_pca=existing_pca,
        w2v2_model_path=w2v2_model_path,
        input_audio_manifests=input_audio_manifests,
        concurrent=featurize_concurrent,
    )

    npy_file, seq_lengths_file, tsv_file = [
        DelayedFormat(f"{{}}/{file_name}", featurize_job.out_features_precompute_pca512_cls128_mean_pooled)
        for file_name in ("train.npy", "train.lengths", "train.tsv")
    ]

    dump_features_to_hdf_job = DumpNumpyFeaturesToHdfJobV2(
        npy_file=npy_file,
        seq_lengths_file=seq_lengths_file,
        tsv_file=tsv_file,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        max_abs_value=max_abs_value,
    )

    cluster_id_file_train, cluster_id_file_valid = [
        DelayedFormat(f"{{}}/CLUS128/{set}.src", featurize_job.out_features) for set in ("train", "valid")
    ]
    cluster_id_file_train = HeadJob(cluster_id_file_train, num_lines="-0").out
    cluster_id_file_valid = HeadJob(cluster_id_file_valid, num_lines="-0").out
    cluster_id_file = ConcatenateJob([cluster_id_file_train, cluster_id_file_valid]).out
    tsv_file_clusters_train, tsv_file_clusters_valid = [
        DelayedFormat(f"{{}}/CLUS128/{file_name}", featurize_job.out_features)
        for file_name in ("train.tsv", "valid.tsv")
    ]
    # remove header line from tsv (this also converts the DelayedFormat to a Path as a nice side effect)
    tsv_path_train = TailJob(tsv_file_clusters_train, num_lines="+2").out
    tsv_path_valid = TailJob(tsv_file_clusters_valid, num_lines="+2").out
    tsv_path = ConcatenateJob([tsv_path_train, tsv_path_valid]).out
    # reformat line "146197/1061-146197-0004.flac    237120" -> "train-other-960/1061-146197-0004/1061-146197-0004"
    reformat_seq_tags = PipelineJob(tsv_path, [rf"sed -E 's|^[^/]*/([^.]+)\.flac.*|{librispeech_key}/\1/\1|'"]).out
    dump_cluster_ids_to_hdf_job = DumpClusterIndicesToHdfJob(
        text_file=cluster_id_file,
        num_clusters=128,
        concurrent=dump_hdf_concurrent,
        fixed_random_subset=fixed_random_subset,
        tsv_file=None,
        seq_tag_file=reformat_seq_tags,
        remove_cluster_repetitions=remove_cluster_repetitions,
    )
    tk.register_output(
        "data/librispeech/audio/featurized/train_other_960.clus128.hdf", dump_cluster_ids_to_hdf_job.out_hdfs[0]
    )

    return (
        list(dump_features_to_hdf_job.out_hdfs.values()),
        featurize_job.out_features_clusters,
        featurize_job.out_features_pca,
        list(dump_cluster_ids_to_hdf_job.out_hdfs.values()),
    )
