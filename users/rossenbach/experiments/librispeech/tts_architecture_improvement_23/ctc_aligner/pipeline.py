from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2

def ctc_training(config, returnn_exe, returnn_root, prefix, num_epochs=100):

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

    return train_job


def ctc_forward(checkpoint, config, returnn_exe, returnn_root, prefix):
    last_forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=[],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        device="cpu",
        cpu_rqmt=8,
        mem_rqmt=8,
    )
    last_forward_job.add_alias(prefix + "/forward")
    alignment_hdf = last_forward_job.out_hdf_files["output.hdf"]
    tk.register_output(prefix + "/training.alignment", alignment_hdf)

    return alignment_hdf


def ctc_search(checkpoint, config, returnn_exe, returnn_root, prefix):
    last_forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=["recognition.txt"],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        device="cpu",
        cpu_rqmt=8,
        mem_rqmt=8,
    )
    last_forward_job.add_alias(prefix + "/forward")
    recognition = last_forward_job.out_hdf_files["recognition.txt"]
    tk.register_output(prefix + "/search.recognition.txt", recognition)
    
    
@tk.block()
def compute_prior(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
):
    """
    Run search for a specific test dataset

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param Path returnn_exe:
    :param Path returnn_root:
    """
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=1,
        device="gpu",
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )
    search_job.add_alias(prefix_name + "/prior_job")
    return search_job.out_files["prior.txt"]
