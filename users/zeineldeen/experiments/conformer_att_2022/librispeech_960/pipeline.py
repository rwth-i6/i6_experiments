from __future__ import annotations
from typing import Optional, Union, Set

from sisyphus import tk
import os.path

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.training import GetBestTFCheckpointJob
from i6_core.returnn.training import AverageTFCheckpointsJob

from .default_tools import SCTK_BINARY_PATH


def training(
    prefix_name, returnn_config, returnn_exe, returnn_root, num_epochs, mem_rqmt=15, time_rqmt=168, gpu_mem=None
):
    """

    :param prefix_name:
    :param returnn_config:
    :param returnn_exe:
    :param returnn_root:
    :return:
    """
    default_rqmt = {
        "mem_rqmt": mem_rqmt,
        "time_rqmt": time_rqmt,
        "log_verbosity": 5,
        "returnn_python_exe": returnn_exe,
        "returnn_root": returnn_root,
    }

    train_job = ReturnnTrainingJob(returnn_config=returnn_config, num_epochs=num_epochs, **default_rqmt)
    if gpu_mem:
        assert gpu_mem in [11, 24]
        train_job.rqmt["gpu_mem"] = gpu_mem
    train_job.add_alias(prefix_name + "/training")
    tk.register_output(prefix_name + "/learning_rates", train_job.out_learning_rates)

    return train_job


def get_best_checkpoint(training_job, key="dev_score_output/output_prob"):
    """
    :param ReturnnTrainingJob training_job:
    :return:
    """
    best_checkpoint_job = GetBestTFCheckpointJob(
        training_job.out_model_dir, training_job.out_learning_rates, key=key, index=0
    )
    return best_checkpoint_job.out_checkpoint


def get_average_checkpoint(
    training_job,
    returnn_exe,
    returnn_root,
    num_average: int = 4,
    key="dev_score_output/output_prob",
):
    """
    get an averaged checkpoint using n models

    :param training_job:
    :param num_average:
    :return:
    """
    epochs = []
    for i in range(num_average):
        best_checkpoint_job = GetBestTFCheckpointJob(
            training_job.out_model_dir,
            training_job.out_learning_rates,
            key=key,
            index=i,
        )
        epochs.append(best_checkpoint_job.out_epoch)
    average_checkpoint_job = AverageTFCheckpointsJob(
        training_job.out_model_dir,
        epochs=epochs,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    return average_checkpoint_job.out_checkpoint


def search_single(
    prefix_name,
    returnn_config,
    checkpoint,
    recognition_dataset,
    recognition_reference,
    recognition_bliss_corpus,
    returnn_exe,
    returnn_root,
    mem_rqmt,
    time_rqmt,
    use_sclite=False,
    use_returnn_compute_wer=True,
    recog_ext_pipeline=False,
    remove_label: Optional[Union[str, Set[str]]] = None,
    gpu_mem: int = 11,
    use_gpu_test: bool = False,
    device: str = "gpu",
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
    :param recog_ext_pipeline: the search output is the raw beam search output, all beams.
        still need to select best, and also still need to maybe remove blank/EOS/whatever.
    :param remove_label: for SearchRemoveLabelJob
    :param gpu_mem: used in sis settings.py to select gpu partition
    :param use_gpu_test: if enabled, use gpu_test partition (usually just for testing)
    :param device: "gpu" or "cpu"
    """
    from i6_core.returnn.search import (
        ReturnnSearchJobV2,
        SearchBPEtoWordsJob,
        ReturnnComputeWERJob,
    )

    search_job = ReturnnSearchJobV2(
        search_data=recognition_dataset.as_returnn_opts(),
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_gzip=recog_ext_pipeline,
        device=device,
    )
    if use_gpu_test:
        search_job.rqmt["gpu_test"] = True
    elif device == "gpu":
        assert gpu_mem in [11, 24]
        search_job.rqmt["gpu_mem"] = gpu_mem
    search_job.add_alias(prefix_name + "/search_job")
    search_bpe = search_job.out_search_file

    # If we need to remove a label, do it early, before the SearchBeamJoinScoresJob,
    # otherwise SearchBeamJoinScoresJob would not have any effect.
    if remove_label:
        from i6_core.returnn.search import SearchRemoveLabelJob

        search_bpe = SearchRemoveLabelJob(search_bpe, remove_label=remove_label, output_gzip=True).out_search_results

    if recog_ext_pipeline:
        # TODO check if SearchBeamJoinScoresJob makes sense.
        #   results are inconsistent.
        #   one potential explanation: the amount of merges per hyp is uneven, and maybe bad hyps have actual more
        #      entries in the beam due to confusions. then their sum will win over better hyps.
        #   another potential explanation: logsumexp is not correct with length norm.
        #      (btw, with length norm, it's not trivial to correct, as it uses the factor from the whole batch.)
        #   thus, we do not use it for now.
        #   if we would use it, only if there was some remove_label.

        from i6_core.returnn.search import SearchTakeBestJob

        search_bpe = SearchTakeBestJob(search_bpe, output_gzip=True).out_best_search_results

    search_words = SearchBPEtoWordsJob(search_bpe, output_gzip=recog_ext_pipeline).out_word_search_results

    wer_ = None

    if use_sclite:
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
        wer_ = sclite_job.out_wer

    if use_returnn_compute_wer:
        wer = ReturnnComputeWERJob(search_words, recognition_reference)

        tk.register_output(prefix_name + "/search_out_words.py", search_words)
        tk.register_output(prefix_name + "/wer", wer.out_wer)
        wer_ = wer.out_wer

    return wer_, search_words


def search(
    prefix_name,
    returnn_config,
    checkpoint,
    test_dataset_tuples,
    returnn_exe,
    returnn_root,
    mem_rqmt=8,
    time_rqmt=1,
    use_sclite=False,
    recog_ext_pipeline=False,
    remove_label: Optional[Union[str, Set[str]]] = None,
    enable_mail: bool = False,
):
    """

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param test_dataset_tuples:
    :param returnn_exe:
    :param returnn_root:
    :param recog_ext_pipeline: the search output is the raw beam search output, all beams.
        still need to select best, and also still need to maybe remove blank/EOS/whatever.
    :param remove_label: for SearchRemoveLabelJob
    :return:
    """
    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    for key, (test_dataset, test_dataset_reference, test_bliss_corpus) in test_dataset_tuples.items():
        wers[key], _ = search_single(
            prefix_name + "/%s" % key,
            returnn_config,
            checkpoint,
            test_dataset,
            test_dataset_reference,
            test_bliss_corpus,
            returnn_exe,
            returnn_root,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
            use_sclite=use_sclite,
            recog_ext_pipeline=recog_ext_pipeline,
            remove_label=remove_label,
        )

    from i6_core.report import GenerateReportStringJob, MailJob

    format_string_report = ",".join(["{%s_val}" % key for key in test_dataset_tuples.keys()])
    format_string = " - ".join(["{%s}: {%s_val}" % (key, key) for key in test_dataset_tuples.keys()])
    values = {}
    values_report = {}
    for key in test_dataset_tuples.keys():
        wer_ = wers[key]
        if wer_ is None:
            continue
        values[key] = key
        values["%s_val" % key] = wers[key]
        values_report["%s_val" % key] = wers[key]

    if not values:
        return None

    report = GenerateReportStringJob(report_values=values, report_template=format_string, compress=False).out_report
    if enable_mail:
        mail = MailJob(result=report, subject=prefix_name, send_contents=True).out_status
        tk.register_output(os.path.join(prefix_name, "mail_status"), mail)
    return format_string_report, values_report
