from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2

def tts_training(config, returnn_exe, returnn_root, prefix, num_epochs=100):

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
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/training.models", train_job.out_model_dir)
    tk.register_output(prefix + "/training.lr", train_job.out_learning_rates)

    return train_job


def tts_forward(checkpoint, config, returnn_exe, returnn_root, prefix):
    last_forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=["corpus.xml.gz"],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        device="cpu",
        cpu_rqmt=8,
        mem_rqmt=16,
    )
    last_forward_job.add_alias(prefix + "/forward")
    corpus = last_forward_job.out_hdf_files["corpus.xml.gz"]
    tk.register_output(prefix + "/fake_corpus.xml.gz", corpus)
    return corpus


def tts_forward_v2(checkpoint, config, returnn_exe, returnn_root, prefix):
    last_forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=config,
        output_files=["corpus.xml.gz", "audio_out"],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        device="cpu",
        cpu_rqmt=8,
        mem_rqmt=16,
    )
    last_forward_job.add_alias(prefix + "/forward")
    corpus = last_forward_job.out_files["corpus.xml.gz"]
    tk.register_output(prefix + "/fake_corpus.xml.gz", corpus)
    return corpus