from sisyphus import tk
import copy
import os
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from i6_core.returnn.search import SearchBPEtoWordsJob

from i6_experiments.users.rossenbach.tts.evaluation.nisqa import NISQAMosPredictionJob

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset

from i6_experiments.users.rilling.evaluation.jobs.hdf_mean import MeanHDFContentJob

from .default_tools import SCTK_BINARY_PATH, NISQA_REPO

def training(config, returnn_exe, returnn_root, prefix, num_epochs=65):
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


def forward(
    checkpoint,
    config,
    returnn_exe,
    returnn_root,
    prefix,
    alias_addition=None,
    target="audio",
    extra_evaluation_epoch=None,
    joint_data=False,
):
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
        device="cpu"
    )

    # last_forward_job.rqmt["gpu_mem"] = 24

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


@tk.block()
def search_single(
    prefix_name,
    returnn_config,
    checkpoint,
    recognition_dataset: GenericDataset,
    recognition_bliss_corpus,
    returnn_exe,
    returnn_root,
    mem_rqmt=8,
):
    """
    Run search for a specific test dataset

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param returnn_standalone.data.datasets.dataset.GenericDataset recognition_dataset:
    :param Path recognition_reference: Path to a py-dict format reference file
    :param Path returnn_exe:
    :param Path returnn_root:
    """
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward"] = recognition_dataset.as_returnn_opts()
    search_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        hdf_outputs=["search_out.py"],
        device="cpu",
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_words = SearchBPEtoWordsJob(search_job.out_hdf_files["search_out.py"]).out_word_search_results

    from i6_core.returnn.search import SearchWordsToCTMJob
    from i6_core.corpus.convert import CorpusToStmJob
    from i6_core.recognition.scoring import ScliteJob

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=search_words,
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer


@tk.block()
def search(prefix_name, returnn_config, checkpoint, test_dataset_tuples, returnn_exe, returnn_root):
    """

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param test_dataset_tuples:
    :param returnn_exe:
    :param returnn_root:
    :return:
    """
    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        wers[key] = search_single(
            prefix_name + "/%s" % key,
            returnn_config,
            checkpoint,
            test_dataset,
            test_dataset_reference,
            returnn_exe,
            returnn_root,
        )

    from i6_core.report import GenerateReportStringJob

    clean_prefix_name = prefix_name.replace(".", "_")
    format_string_report = ",".join(["{%s_val}" % (clean_prefix_name + key) for key in test_dataset_tuples.keys()])
    format_string = " - ".join(
        ["{%s}: {%s_val}" % (clean_prefix_name + key, clean_prefix_name + key) for key in test_dataset_tuples.keys()]
    )
    values = {}
    values_report = {}
    for key in test_dataset_tuples.keys():
        values[clean_prefix_name + key] = key
        values["%s_val" % (clean_prefix_name + key)] = wers[key]
        values_report["%s_val" % (clean_prefix_name + key)] = wers[key]

    report = GenerateReportStringJob(report_values=values, report_template=format_string, compress=False).out_report
    tk.register_output(os.path.join(prefix_name, "report"), report)
    return format_string_report, values_report



# def evaluate_invertibility(name, checkpoint, forward_config, returnn_root, returnn_exe):
#     forward_job = forward(
#         checkpoint=checkpoint,
#         config=forward_config,
#         returnn_exe=returnn_exe,
#         returnn_root=returnn_root,
#         prefix=name,
#         target="invertibility",
#     )

#     calc_mean_job = MeanHDFContentJob(forward_job.out_hdf_files["output.hdf"])

#     tk.register_output(name + "/invertibility", calc_mean_job.out_mean)

#     return forward_job
