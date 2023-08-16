from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob

def glowTTS_training(config, returnn_exe, returnn_root, prefix, num_epochs=65):

    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=100,
        mem_rqmt=16,
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    if config.get("import_model_train_epoch1", False):
        train_job.add_alias(prefix + "/training2")
        tk.register_output(prefix + "/training2.models", train_job.out_model_dir)
    else:
        train_job.add_alias(prefix + "/training")
        tk.register_output(prefix + "/training.models", train_job.out_model_dir)
    # tk.register_output(prefix + "/training.learning_rates", train_job.out_learning_rates)

    return train_job

def glowTTS_forward(checkpoint, config, returnn_exe, returnn_root, prefix, alias_addition=None):
    last_forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=[],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        mem_rqmt=8
    )
    tts_hdf = last_forward_job.out_hdf_files["output.hdf"]
    
    if alias_addition is None:
        last_forward_job.add_alias(prefix + "/forward")
        tk.register_output(prefix + "/forward.spectograms", tts_hdf)
    else:
        last_forward_job.add_alias(prefix + "/forward" + alias_addition)
        tk.register_output(prefix + "/forward" + alias_addition + ".spectograms", tts_hdf)

    return tts_hdf
