from sisyphus import tk
import copy
import os
from i6_core.returnn.forward import ReturnnForwardJob, ReturnnForwardJobV2
from i6_core.returnn.search import SearchBPEtoWordsJob

from i6_experiments.users.rossenbach.tts.evaluation.nisqa import NISQAMosPredictionJob

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset

from .default_tools import SCTK_BINARY_PATH, NISQA_REPO

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


def tts_eval(
    prefix_name,
    returnn_config,
    checkpoint,
    returnn_exe,
    returnn_exe_asr,
    returnn_root,
    mem_rqmt=12,
    vocoder="univnet",
    nisqa_eval=False,
    swer_eval=False,
    swer_eval_corpus_key="train-clean",
    nisqa_confidence=False
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
        time_rqmt=2,
        device="cpu",
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["audio_files", "out_corpus.xml.gz"],
    )

    name = prefix_name + f"/tts_eval_{vocoder}/{swer_eval_corpus_key}"
    forward_job.add_alias(name + "/forward")
    if nisqa_eval:
        evaluate_nisqa(name, forward_job.out_files["out_corpus.xml.gz"], vocoder=vocoder, with_bootstrap=nisqa_confidence)
    if swer_eval:
        evaluate_swer(name, forward_job, returnn_exe=returnn_exe_asr, returnn_root=returnn_root, corpus_key=swer_eval_corpus_key)
    return forward_job


def evaluate_nisqa(prefix_name: str, bliss_corpus: tk.Path, vocoder: str = "univnet", with_bootstrap=False):
    predict_mos_job = NISQAMosPredictionJob(bliss_corpus, nisqa_repo=NISQA_REPO)
    predict_mos_job.add_alias(prefix_name + f"/nisqa_mos")
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/average"), predict_mos_job.out_mos_average)
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/min"), predict_mos_job.out_mos_min)
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/max"), predict_mos_job.out_mos_max)
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/std_dev"), predict_mos_job.out_mos_std_dev)

    if with_bootstrap:
        from i6_experiments.users.rossenbach.tts.evaluation.nisqa import NISQAConfidenceJob
        nisqa_confidence_job = NISQAConfidenceJob(predict_mos_job.output_dir, bliss_corpus)
        nisqa_confidence_job.add_alias(prefix_name + "/nisqa_mos_confidence")
        tk.register_output(os.path.join(prefix_name, "nisqa_mos/confidence_max_interval"), nisqa_confidence_job.out_max_interval_bound)


def evaluate_swer(
    name: str,
    forward_job: tk.Job,
    returnn_exe,
    returnn_root,
    corpus_key,
    # synthetic_bliss: tk.Path,
    # system: ASRRecognizerSystem,
    # with_confidence=False,
):
    asr_system = "ls960eow_phon_ctc_50eps_fastsearch"

    from i6_experiments.users.rossenbach.experiments.jaist_project.storage import asr_recognizer_systems
    from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob, MergeStrategy
    from .data import build_swer_test_dataset, get_bliss_corpus_dict, get_tts_lexicon

    synthetic_bliss = MergeCorporaWithPathResolveJob(
        bliss_corpora=[forward_job.out_files["out_corpus.xml.gz"]],
        name=corpus_key,  # important to keep the original sequence names for matching later
        merge_strategy=MergeStrategy.FLAT,
    ).out_merged_corpus
    system = asr_recognizer_systems[asr_system]

    # bliss_corpus, _ = get_tts_eval_bliss_and_zip(
    #     ls_corpus_key=corpus_key, silence_preprocessed=False, remove_unk_seqs=True
    # )

    from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

    bliss_corpus = FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=get_bliss_corpus_dict()[corpus_key], bliss_lexicon=get_tts_lexicon(), all_unknown=False
    ).out_corpus

    search_single(
        prefix_name=name + "/swer/" + asr_system,
        returnn_config=system.config,
        checkpoint=system.checkpoint,
        recognition_dataset=build_swer_test_dataset(
            synthetic_bliss=synthetic_bliss,
            returnn_exe=returnn_exe, 
            returnn_root=returnn_root,
            preemphasis=system.preemphasis,
            peak_normalization=system.peak_normalization,
        ),
        # recognition_bliss_corpus=bliss_corpus,
        recognition_bliss_corpus=bliss_corpus,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        # with_confidence=with_confidence,
    )
