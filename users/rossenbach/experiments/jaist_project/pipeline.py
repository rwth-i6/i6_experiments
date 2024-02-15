"""
Pipeline parts to create the necessary jobs for training / forwarding / search etc...
"""
import copy
import os.path
from typing import Any, Dict, Optional, Tuple

from sisyphus import tk

from i6_core.corpus.convert import CorpusToStmJob, CorpusReplaceOrthFromReferenceCorpus
from i6_core.corpus.transform import MergeStrategy
from i6_core.recognition.scoring import ScliteJob
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob
from i6_experiments.users.rossenbach.tts.evaluation.nisqa import NISQAMosPredictionJob

from .config import get_forward_config, get_training_config
from .data.aligner import build_training_dataset
from .data.tts_phon import get_tts_extended_bliss, build_fixed_speakers_generating_dataset, build_durationtts_training_dataset
from .default_tools import SCTK_BINARY_PATH, NISQA_REPO, RETURNN_EXE, MINI_RETURNN_ROOT
from .storage import add_synthetic_data


@tk.block()
def training(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        num_epochs: int,
) -> ReturnnTrainingJob:
    """
    Perform RETURNN training

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for training
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param num_epochs: for how many epochs to train
    :return: training job
    """
    default_rqmt = {
        'mem_rqmt': 32,
        'time_rqmt': 168,
        'cpu_rqmt': 16,
        'log_verbosity': 5,
        'returnn_python_exe': returnn_exe,
        'returnn_root': returnn_root,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        num_epochs=num_epochs,
        **default_rqmt
    )
    train_job.add_alias(prefix_name + "/training")
    tk.register_output(prefix_name + "/learning_rates", train_job.out_learning_rates)

    return train_job


def search_single(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        recognition_dataset: Dataset,
        recognition_bliss_corpus: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: float = 8,
        use_gpu:bool = False,
):
    """
    Run search for a specific test dataset

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param recognition_dataset: Dataset to perform recognition on
    :param recognition_bliss_corpus: path to bliss file used as Sclite evaluation reference
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: some search jobs might need more memory
    :param use_gpu: if to do GPU decoding
    """
    returnn_config = copy.deepcopy(returnn_config)
    returnn_config.config["forward"] = recognition_dataset.as_returnn_opts()
    search_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=24,
        device="gpu" if use_gpu else "cpu",
        cpu_rqmt=2,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py"],
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=search_job.out_files["search_out.py"],
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer, search_job


def run_swer_evaluation(
        prefix_name: str,
        asr_returnn_config: ReturnnConfig,
        asr_checkpoint: tk.Path,
        synthetic_bliss: tk.Path,
        preemphasis: float,
        peak_normalization: bool,
    ):
    from .data.common import build_swer_test_dataset, get_cv_bliss
    search_single(
        prefix_name=prefix_name,
        returnn_config=asr_returnn_config,
        checkpoint=asr_checkpoint,
        recognition_dataset=build_swer_test_dataset(
            synthetic_bliss=synthetic_bliss,
            preemphasis=preemphasis,
            peak_normalization=peak_normalization
        ),
        recognition_bliss_corpus=get_cv_bliss(),
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT
    )


@tk.block()
def search(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        test_dataset_tuples,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        use_gpu: bool = False
):
    """
    Run search over multiple datasets and collect statistics

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param test_dataset_tuples: tuple of (Dataset, tk.Path) for the dataset object and the reference bliss
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :return:
    """
    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        wers[key], search_job = search_single(prefix_name + "/%s" % key, returnn_config, checkpoint, test_dataset, test_dataset_reference, returnn_exe, returnn_root, use_gpu=use_gpu)
        search_jobs.append(search_job)

    from i6_core.report import GenerateReportStringJob, MailJob
    format_string_report = ",".join(["{%s_val}" % (prefix_name + key) for key in test_dataset_tuples.keys()])
    format_string = " - ".join(["{%s}: {%s_val}" % (prefix_name + key, prefix_name + key) for key in test_dataset_tuples.keys()])
    values = {}
    values_report = {}
    for key in test_dataset_tuples.keys():
        values[prefix_name + key] = key
        values["%s_val" % (prefix_name + key)] = wers[key]
        values_report["%s_val" % (prefix_name + key)] = wers[key]

    report = GenerateReportStringJob(report_values=values, report_template=format_string, compress=False).out_report
    #mail = MailJob(result=report, subject=prefix_name, send_contents=True).out_status
    #tk.register_output(os.path.join(prefix_name, "mail_status"), mail)
    return format_string_report, values_report, search_jobs


@tk.block()
def compute_prior(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: int = 8,
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


@tk.block()
def extract_durations(
        prefix_name: str,
        returnn_config: ReturnnConfig,
        checkpoint: tk.Path,
        returnn_exe: tk.Path,
        returnn_root: tk.Path,
        mem_rqmt: int = 8,
):
    """
    Run phoneme duration extraction

    :param prefix_name: prefix folder path for alias and output files
    :param returnn_config: the RETURNN config to be used for forwarding
    :param Checkpoint checkpoint: path to RETURNN PyTorch model checkpoint
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param mem_rqmt: override the default memory requirement
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
        output_files=["durations.hdf"],
    )
    search_job.add_alias(prefix_name + "/duration_extraction_job")
    return search_job.out_files["durations.hdf"]


@tk.block()
def tts_eval(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
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
        output_files=["audio_files"],
    )
    forward_job.add_alias(prefix_name + "/tts_eval_job")
    return forward_job


def tts_eval_v2(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=8,
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
    forward_job.add_alias(prefix_name + "/tts_eval_job")
    evaluate_nisqa(prefix_name, forward_job.out_files["out_corpus.xml.gz"])
    return forward_job


@tk.block()
def tts_generation(
        prefix_name,
        returnn_config,
        checkpoint,
        returnn_exe,
        returnn_root,
        mem_rqmt=32,
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
        time_rqmt=24,
        device="cpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["audio_files", "out_corpus.xml.gz"],
    )
    forward_job.add_alias(prefix_name + "/tts_eval_job")
    return forward_job


def generate_synthetic(
        prefix: str,
        name: str,
        target_ls_corpus_key: str,
        checkpoint: tk.Path,
        params: Dict[str, Any],
        net_module: str,
        decoder_options: Dict[str, Any],
        extra_decoder: Optional[str] = None,
        extra_forward_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        splits: int = 10,
        randomize_speaker: bool = True,
        use_subset=False,
):
    """
    use a TTS system to create a synthetic corpus
    """
    # we want to get ls360 but with the vocab settings from ls100
    asr_bliss = get_bliss_corpus_dict()[target_ls_corpus_key]
    tts_bliss = get_tts_extended_bliss(ls_corpus_key=target_ls_corpus_key, lexicon_ls_corpus_key=target_ls_corpus_key)

    if use_subset:
        from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
        from i6_core.corpus.filter import FilterCorpusBySegmentsJob
        # from i6_core.text.processing import
        segments = SegmentCorpusJob(asr_bliss,1).out_single_segment_files[1]
        segment_split = ShuffleAndSplitSegmentsJob(
            segment_file=segments,
            split={"100-from-360": 0.2743765262368527, "rest": 0.7256234737631473}
        )
        sub100_segments = segment_split.out_segments["100-from-360"]
        asr_bliss = FilterCorpusBySegmentsJob(
            bliss_corpus=asr_bliss,
            segment_file=sub100_segments,
            compressed=True,
            delete_empty_recordings=True
        ).out_corpus
        tts_bliss = FilterCorpusBySegmentsJob(
            bliss_corpus=tts_bliss,
            segment_file=sub100_segments,
            compressed=True,
            delete_empty_recordings=True
        ).out_corpus

    generating_datasets = build_fixed_speakers_generating_dataset(
        text_bliss=tts_bliss,
        num_splits=splits,
        ls_corpus_key="train-clean-100",  # this is always ls100
        randomize_speaker=randomize_speaker,
    )
    split_out_bliss = []
    for i in range(splits):
        forward_config = get_forward_config(
            network_module=net_module,
            net_args=params,
            decoder=extra_decoder or net_module,
            decoder_args=decoder_options,
            config={
                "forward": generating_datasets.split_datasets[i].as_returnn_opts()
            },
            debug=debug,
        )
        if extra_forward_config is not None:
            forward_config.config.update(extra_forward_config)
        forward_job = tts_generation(
            prefix_name=prefix + name + f"/{target_ls_corpus_key + ('-sub100' if use_subset else '')}_split{i}",
            returnn_config=forward_config,
            checkpoint=checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        split_out_bliss.append(forward_job.out_files["out_corpus.xml.gz"])

    merged_corpus = MergeCorporaWithPathResolveJob(
        bliss_corpora=split_out_bliss, name=target_ls_corpus_key, merge_strategy=MergeStrategy.FLAT
    ).out_merged_corpus
    merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=merged_corpus,
        reference_bliss_corpus=asr_bliss,
    ).out_corpus
    ogg_zip_job = BlissToOggZipJob(
        merged_corpus_with_text,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT
    )
    ogg_zip_job.add_alias(prefix + name + f"/{target_ls_corpus_key + ('-sub100' if use_subset else '')}/create_synthetic_zip")
    add_synthetic_data(name + "_" + target_ls_corpus_key + ("-sub100" if use_subset else ""), ogg_zip_job.out_ogg_zip, merged_corpus_with_text)
    return merged_corpus_with_text


def cross_validation_nisqa(prefix, name, params, net_module, checkpoint, decoder_options, extra_decoder=None, debug=False):
    training_datasets = build_training_dataset()
    forward_config = get_forward_config(
        network_module=net_module,
        net_args=params,
        decoder=extra_decoder or net_module,
        decoder_args=decoder_options,
        config={
            "forward": training_datasets.cv.as_returnn_opts()
        },
        debug=debug,
    )
    forward_job = tts_eval_v2(
        prefix_name=prefix + name,
        returnn_config=forward_config,
        checkpoint=checkpoint,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
    return forward_job


def evaluate_nisqa(
        prefix_name: str,
        bliss_corpus: tk.Path,
):
    predict_mos_job = NISQAMosPredictionJob(bliss_corpus, nisqa_repo=NISQA_REPO)
    predict_mos_job.add_alias(prefix_name + "/nisqa_mos")
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/average"), predict_mos_job.out_mos_average)
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/min"), predict_mos_job.out_mos_min)
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/max"), predict_mos_job.out_mos_max)
    tk.register_output(os.path.join(prefix_name, "nisqa_mos/std_dev"), predict_mos_job.out_mos_std_dev)


def tts_training(
        prefix: str,
        name: str,
        params: Dict[str, Any],
        net_module: str,
        config: Dict[str, Any],
        duration_hdf: tk.Path,
        decoder_options: Dict[str, Any],
        extra_decoder: Optional[str] = None,
        use_custom_engine:bool = False,
        debug: bool = False,
        num_epochs=200
) -> Tuple[ReturnnTrainingJob, ReturnnForwardJobV2]:
    """

    :param prefix:
    :param name:
    :param params:
    :param net_module:
    :param config:
    :param duration_hdf:
    :param decoder_options:
    :param extra_decoder:
    :param use_custom_engine:
    :param debug:
    :param num_epochs:
    :return:
    """
    training_datasets = build_durationtts_training_dataset(duration_hdf=duration_hdf)
    training_config = get_training_config(
        training_datasets=training_datasets,
        network_module=net_module,
        net_args=params,
        config=config,
        debug=debug,
        use_custom_engine=use_custom_engine,
    )  # implicit reconstruction loss
    forward_config = get_forward_config(
        network_module=net_module,
        net_args=params,
        decoder=extra_decoder or net_module,
        decoder_args=decoder_options,
        config={
            "forward": training_datasets.cv.as_returnn_opts()
        },
        debug=debug,
    )
    train_job = training(
        prefix_name=prefix + name,
        returnn_config=training_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        num_epochs=num_epochs
    )
    forward_job = tts_eval_v2(
        prefix_name=prefix + name,
        returnn_config=forward_config,
        checkpoint=train_job.out_checkpoints[num_epochs],
        returnn_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
    return train_job, forward_job

