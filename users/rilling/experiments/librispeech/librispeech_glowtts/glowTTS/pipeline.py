from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob

def glowTTS_training(config, returnn_exe, returnn_root, prefix, num_epochs=65):

    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=100,
        mem_rqmt=10,
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
)
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/training.models", train_job.out_model_dir)

    return train_job

def glowTTS_forward(checkpoint, config, returnn_exe, returnn_root, prefix, alias_addition=None, target="audio", extra_evaluation_epoch=None, joint_data=False):
    hdf_outputs = [] if target != "audio" else ["/var/tmp/lukas.rilling/out"]
    if target == "audio":
        hdf_outputs = ["/var/tmp/lukas.rilling/out"]
    elif target == "latent_space":
        hdf_outputs = ["samples.hdf", "mean.hdf"]
        # hdf_outputs = ["samples.hdf"]
    else:
        hdf_outputs = []

    last_forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=hdf_outputs,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        mem_rqmt=20
    )

    if (target == "spectrogram" and joint_data):
        last_forward_job.rqmt["gpu_mem"] = 24

    forward_prefix = prefix + "/forward"

    if target != "audio":
        forward_prefix += f"_{target}"

    if extra_evaluation_epoch is not None:
        forward_prefix += f"_extra_evaluation_{extra_evaluation_epoch}"

    if alias_addition:
        forward_prefix += alias_addition

    forward_suffix = f"/{target}"

    last_forward_job.add_alias(forward_prefix)

    tts_hdf = None

    if target == "audio":
        tts_hdf = last_forward_job.out_hdf_files["/var/tmp/lukas.rilling/out"]
        tk.register_output(forward_prefix + forward_suffix, tts_hdf)
    elif target == "latent_space":
        samples_hdf = last_forward_job.out_hdf_files["samples.hdf"]
        mean_hdf = last_forward_job.out_hdf_files["mean.hdf"]
        tk.register_output(forward_prefix + forward_suffix + "/samples", samples_hdf)
        tk.register_output(forward_prefix + forward_suffix + "/mean", mean_hdf)
    else:
        tts_hdf = last_forward_job.out_hdf_files["output.hdf"]
        tk.register_output(forward_prefix + forward_suffix, tts_hdf)

    return last_forward_job
