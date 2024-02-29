import os
from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from i6_experiments.users.rossenbach.tts.evaluation.nisqa import NISQAMosPredictionJob

from ..default_tools import NISQA_REPO

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

def glowTTS_forward(checkpoint, config, returnn_exe, returnn_root, prefix, alias_addition=None, target="audio", extra_evaluation_epoch=None, joint_data=False, device="gpu"):
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
        mem_rqmt=20,
        device=device
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


def tts_eval(
    prefix_name,
    returnn_config,
    checkpoint,
    returnn_exe,
    returnn_root,
    mem_rqmt=12,
    vocoder="univnet"
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
    """
    forward_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=1,
        device="cpu",
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["audio_files", "out_corpus.xml.gz"],
    )
    forward_job.add_alias(prefix_name + f"/tts_eval_{vocoder}/forward")
    evaluate_nisqa(prefix_name, forward_job.out_files["out_corpus.xml.gz"], vocoder=vocoder)
    return forward_job


def evaluate_nisqa(
    prefix_name: str,
    bliss_corpus: tk.Path,
    vocoder: str = "univnet"
):
    predict_mos_job = NISQAMosPredictionJob(bliss_corpus, nisqa_repo=NISQA_REPO)
    predict_mos_job.add_alias(prefix_name + f"/tts_eval_{vocoder}/nisqa_mos")
    tk.register_output(os.path.join(prefix_name, f"tts_eval_{vocoder}/nisqa_mos/average"), predict_mos_job.out_mos_average)
    tk.register_output(os.path.join(prefix_name, f"tts_eval_{vocoder}/nisqa_mos/min"), predict_mos_job.out_mos_min)
    tk.register_output(os.path.join(prefix_name, f"tts_eval_{vocoder}/nisqa_mos/max"), predict_mos_job.out_mos_max)
    tk.register_output(os.path.join(prefix_name, f"tts_eval_{vocoder}/nisqa_mos/std_dev"), predict_mos_job.out_mos_std_dev)
