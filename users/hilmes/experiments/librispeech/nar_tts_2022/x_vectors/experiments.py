from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob

from .data import get_xvector_data
from .pipeline import get_training_config, training, get_forward_config, forward


def baseline_xvectors():
    name = f"experiments/librispeech/nar_tts_2022/tts/xvectors/"

    embeddings = {}

    for output_embedding in [256, 512]:
        experiment_name = name + "baseline_" + str(output_embedding)
        returnn_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn", commit="2c0bf3666e721b86d843f2ef54cd416dfde20566"
        ).out_repository
        returnn_root.hash_overwrite = "RETURNN_XVECTORS"
        returnn_exe = tk.Path(
            "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
            hash_overwrite="GENERIC_RETURNN_LAUNCHER",
        )

        training_datasets = get_xvector_data(
            returnn_root=returnn_root, returnn_exe=returnn_exe, output_path=experiment_name
        )
        train_config = get_training_config(training_datasets, output_size=output_embedding)
        train_job = training(
            config=train_config,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root,
            prefix=experiment_name,
            num_epochs=600,
            mem=8,
        )
        tk.register_output(experiment_name + "/last_checkpoint", train_job.out_checkpoints[600].index_path)

        forward_config = get_forward_config(training_datasets, output_size=output_embedding)
        forward_job = forward(
            train_job.out_checkpoints[600],
            config=forward_config,
            prefix=experiment_name,
            returnn_exe=returnn_exe,
            returnn_root=returnn_root)
        tk.register_output(experiment_name + "/speaker_embedding", forward_job.out_default_hdf)

        embeddings[output_embedding] = forward_job.out_default_hdf
        # TODO: Take best checkpoint?
    return embeddings
