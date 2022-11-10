import os
from sisyphus import tk

from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob


def tts_training(returnn_config, num_epochs, returnn_gpu_exe, returnn_root, output_path, **kwargs):
    """

    :param ReturnnConfig returnn_config:
    :param int num_epochs:
    :param Path returnn_gpu_exe:
    :param Path returnn_root:
    :param str output_path:
    :param kwargs: additional parameters for ReturnnTrainingJob
    :return:
    :rtype: ReturnnTrainingJob
    """

    additional_args = {
        "time_rqmt": 120,
        "mem_rqmt": 16,
        "cpu_rqmt": 4,
        **kwargs,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        log_verbosity=5,
        num_epochs=num_epochs,
        returnn_python_exe=returnn_gpu_exe,
        returnn_root=returnn_root,
        **additional_args
    )
    train_job.add_alias(os.path.join(output_path, "tts_training"))

    tk.register_output(os.path.join(output_path, "tts_training.models"), train_job.out_model_dir)
    tk.register_output(os.path.join(output_path, "tts_training.config"), train_job.out_returnn_config_file)

    return train_job


def tts_forward(returnn_config, checkpoint, returnn_gpu_exe, returnn_root, output_path):
    """

    :param ReturnnConfig returnn_config: returnn config for the `forward` task
    :param Checkpoint checkpoint:
    :param Path returnn_gpu_exe:
    :param Path returnn_root:
    :param str output_path:
    :return: synthesized audio feature hdf
    :rtype: Path
    """
    forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        hdf_outputs=[],
        returnn_python_exe=returnn_gpu_exe,
        returnn_root=returnn_root
    )

    forward_job.add_alias(os.path.join(output_path, "tts_forward"))

    return forward_job.out_hdf_files["output.hdf"]