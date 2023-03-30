from sisyphus import tk
from functools import lru_cache
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    get_alignment_data,
)
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.ctc_align.ctc_pipeline import (
    get_training_config,
    get_forward_config,
    ctc_training,
    ctc_forward,
)


@lru_cache
def get_baseline_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository

    name = "experiments/librispeech/nar_tts_2022/ctc_align/ctc_experiments/baseline"

    training_datasets = get_alignment_data(
        name + "/datasets", returnn_exe=returnn_exe, returnn_root=returnn_root
    )

    aligner_config = get_training_config(
        returnn_common_root=returnn_common_root, training_datasets=training_datasets
    )

    train_job = ctc_training(
        config=aligner_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=name,
    )

    returnn_forward_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="240f119b54d52a4324ab300c301f8e003e0a398c",
    ).out_repository
    returnn_common_forward_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="13df2f5dc409bd0a73224c43a90b9eed7f9c29e9",
        checkout_folder_name="returnn_common",
    ).out_repository
    returnn_forward_root.hash_overwrite = "baseline_ctc_alignment_returnn_forward"
    returnn_common_forward_root.hash_overwrite = "baseline_ctc_alignment_rc_forward"

    aligner_forward_confg = get_forward_config(
        returnn_common_root=returnn_common_forward_root,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams,
    )

    durations_hdf = ctc_forward(
        train_job.out_checkpoints[100],
        aligner_forward_confg,
        returnn_exe,
        returnn_forward_root,
        name,
    )
    return durations_hdf


def get_loss_scale_alignments():
    """
    Different reconstruction loss alignents
    :return: dict with durations_hdfs
    """
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository

    name = "experiments/librispeech/nar_tts_2022/ctc_align/ctc_experiments/loss_scale"

    training_datasets = get_alignment_data(
        name + "/datasets", returnn_exe=returnn_exe, returnn_root=returnn_root
    )
    alignments = {}
    # for loss in [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1]:, REMOVED because of space
    for loss in [0, 1]:
        exp_name = name + f"_{loss}"
        aligner_config = get_training_config(
            returnn_common_root=returnn_common_root,
            training_datasets=training_datasets,
            reconstruction_scale=loss,
        )

        train_job = ctc_training(
            config=aligner_config,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=exp_name,
        )

        returnn_forward_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn",
            commit="240f119b54d52a4324ab300c301f8e003e0a398c",
        ).out_repository
        returnn_common_forward_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn_common",
            commit="13df2f5dc409bd0a73224c43a90b9eed7f9c29e9",
            checkout_folder_name="returnn_common",
        ).out_repository
        returnn_forward_root.hash_overwrite = "baseline_ctc_alignment_returnn_forward"
        returnn_common_forward_root.hash_overwrite = "baseline_ctc_alignment_rc_forward"

        aligner_forward_confg = get_forward_config(
            returnn_common_root=returnn_common_forward_root,
            forward_dataset=training_datasets.joint,
            datastreams=training_datasets.datastreams,
        )

        durations_hdf = ctc_forward(
            train_job.out_checkpoints[100],
            aligner_forward_confg,
            returnn_exe,
            returnn_forward_root,
            exp_name,
        )
        alignments[str(loss)] = durations_hdf

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    returnn_common_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="79876b18552f61a3af7c21c670475fee51ef3991",
        checkout_folder_name="returnn_common",
    ).out_repository
    exp_name = name + f"test"
    aligner_config = get_training_config(
        returnn_common_root=returnn_common_root,
        training_datasets=training_datasets,
        reconstruction_scale=1,
    )

    train_job = ctc_training(
        config=aligner_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=exp_name,
    )

    returnn_forward_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="240f119b54d52a4324ab300c301f8e003e0a398c",
    ).out_repository
    returnn_common_forward_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn_common",
        commit="13df2f5dc409bd0a73224c43a90b9eed7f9c29e9",
        checkout_folder_name="returnn_common",
    ).out_repository
    returnn_forward_root.hash_overwrite = "baseline_ctc_alignment_returnn_forward"
    returnn_common_forward_root.hash_overwrite = "baseline_ctc_alignment_rc_forward"

    aligner_forward_confg = get_forward_config(
        returnn_common_root=returnn_common_forward_root,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams,
    )

    durations_hdf = ctc_forward(
        train_job.out_checkpoints[100],
        aligner_forward_confg,
        returnn_exe,
        returnn_forward_root,
        exp_name,
    )
    alignments[str(23)] = durations_hdf
    return alignments
