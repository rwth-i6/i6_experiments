"""
Pipeline parts to create the necessary jobs for training / forwarding / search etc...
"""
import copy
import enum
from dataclasses import dataclass, asdict
import os.path
from typing import Any, Dict, List, Optional, Tuple, Union

from sisyphus import tk

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnTrainingJob, AverageTorchCheckpointsJob, GetBestPtCheckpointJob
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datasets import Dataset

from .config import get_forward_config, get_training_config, get_prior_config, TrainingDatasets, serialize_forward
from .default_tools import SCTK_BINARY_PATH, RETURNN_EXE, MINI_RETURNN_ROOT


@dataclass
class ASRModel:
    checkpoint: tk.Path
    net_args: Dict[str, Any]
    network_module: str
    prior_file: Optional[tk.Path]
    prefix_name: Optional[str]


def search_single(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    recognition_dataset: Dataset,
    recognition_bliss_corpus: tk.Path,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    mem_rqmt: float = 10,
    use_gpu: bool = False,
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
        cpu_rqmt=8 if mem_rqmt < 30 else 16,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["search_out.py"],
    )
    search_job.add_alias(prefix_name + "/search_job")

    search_ctm_job = SearchWordsToCTMJob(
        recog_words_file=search_job.out_files["search_out.py"],
        bliss_corpus=recognition_bliss_corpus,
    )
    search_ctm_job.add_alias(prefix_name + "/ctm_job")
    search_ctm = search_ctm_job.out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=3)
    sclite_job.add_alias(prefix_name + "/sclite_job")
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer, search_job


@tk.block()
def search(
    prefix_name: str,
    forward_config: Dict[str, Any],
    asr_model: ASRModel,
    decoder_module: str,
    decoder_args: Dict[str, Any],
    test_dataset_tuples: Dict[str, Tuple[Dataset, tk.Path]],
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    use_gpu: bool = False,
    import_memristor: bool = False,
    debug: bool = False,
):
    """
    Run search over multiple datasets and collect statistics

    :param prefix_name: prefix folder path for alias and output files
    :param forward_config: returnn config parameter for the forward job
    :param asr_model: the ASRModel from the training
    :param decoder_module: path to the file containing the decoder definition
    :param decoder_args: arguments for the decoding forward_init_hook
    :param test_dataset_tuples: tuple of (Dataset, tk.Path) for the dataset object and the reference bliss
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    :param use_gpu: run search with GPU
    """
    if asr_model.prior_file is not None:
        decoder_args["config"]["prior_file"] = asr_model.prior_file

    returnn_search_config = get_forward_config(
        network_module=asr_model.network_module,
        config=forward_config,
        net_args=asr_model.net_args,
        decoder_args=decoder_args,
        decoder=decoder_module,
        debug=debug,
        import_memristor=import_memristor,
    )

    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        search_name = prefix_name + "/%s" % key
        if "hubert_tune" in search_name:
            mem = 30
        elif "RelPosEnc" in search_name:
            mem = 16
        else:
            mem = 10
        wers[search_name], search_job = search_single(
            search_name,
            returnn_search_config,
            asr_model.checkpoint,
            test_dataset,
            test_dataset_reference,
            returnn_exe,
            returnn_root,
            mem_rqmt=mem,
            use_gpu=use_gpu,
        )
        search_jobs.append(search_job)

    return search_jobs, wers


@tk.block()
def compute_prior(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    mem_rqmt: int = 16,
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
        time_rqmt=2,
        device="gpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["prior.txt"],
    )
    if "hubert_tune_v1_xlarge" in prefix_name or "hubert_tune_v2_xlarge" in prefix_name:
        search_job.rqmt["time"] += 12
    search_job.add_alias(prefix_name + "/prior_job")
    return search_job.out_files["prior.txt"]


@tk.block()
def quantize_static(
    prefix_name: str,
    returnn_config: ReturnnConfig,
    checkpoint: tk.Path,
    returnn_exe: tk.Path,
    returnn_root: tk.Path,
    mem_rqmt: int = 16,
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
    quantize_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=2,
        device="gpu",
        cpu_rqmt=8,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        output_files=["model.pt", "seq_tags.txt"],
    )
    quantize_job.set_keep_value(5)
    quantize_job.add_alias(prefix_name + "/calibration")
    return quantize_job.out_files["model.pt"]


def training(training_name, datasets, train_args, num_epochs, returnn_exe, returnn_root):
    """
    :param training_name:
    :param datasets:
    :param train_args:
    :param num_epochs:
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    """
    returnn_config = get_training_config(training_datasets=datasets, **train_args)
    default_rqmt = {
        "mem_rqmt": 24,
        "time_rqmt": 168,
        "cpu_rqmt": 6,
        "log_verbosity": 5,
        "returnn_python_exe": returnn_exe,
        "returnn_root": returnn_root,
    }

    train_job = ReturnnTrainingJob(returnn_config=returnn_config, num_epochs=num_epochs, **default_rqmt)
    train_job.add_alias(training_name + "/training")
    tk.register_output(training_name + "/learning_rates", train_job.out_learning_rates)
    return train_job


def prepare_asr_model(
    training_name,
    train_job,
    train_args,
    with_prior,
    datasets: Optional[TrainingDatasets] = None,
    get_specific_checkpoint: Optional[int] = None,
    get_best_averaged_checkpoint: Optional[Tuple[int, str]] = None,
    get_last_averaged_checkpoint: Optional[int] = None,
    prior_config: Optional[Dict[str, Any]] = None,
):
    """
    :param training_name:
    :param train_job: output of training
    :param train_args: same args as for training
    :param with_prior: If prior should be used (yes for CTC, no for RNN-T)
    :param datasets: Needed if with_prior == True
    :param get_specific_checkpoint: return a specific epoch (set one get_*)
    :param get_best_averaged_checkpoint: return the average with (n checkpoints, loss-key), n checkpoints can be 1
    :param get_last_averaged_checkpoint: return the average of the last n checkpoints
    :param prior_config: if with_prior is true, can be used to add Returnn config parameters for the prior compute job
    :return:
    """

    params = [get_specific_checkpoint, get_last_averaged_checkpoint, get_best_averaged_checkpoint]
    assert sum([p is not None for p in params]) == 1
    assert not with_prior or datasets is not None

    if get_best_averaged_checkpoint is not None:
        num_checkpoints, loss_key = get_best_averaged_checkpoint
        checkpoints = []
        for index in range(num_checkpoints):
            best_job = GetBestPtCheckpointJob(
                train_job.out_model_dir,
                train_job.out_learning_rates,
                key=loss_key,
                index=index,
            )
            best_job.add_alias(training_name + f"/get_best_job_{index}")
            checkpoints.append(best_job.out_checkpoint)
        if num_checkpoints > 1:
            # perform averaging
            avg = AverageTorchCheckpointsJob(
                checkpoints=checkpoints, returnn_python_exe=RETURNN_EXE, returnn_root=MINI_RETURNN_ROOT
            )
            avg.keep_value(15)
            if "v6_20_1024" in training_name:
                avg.rqmt["mem"] += 2
            checkpoint = avg.out_checkpoint
            avg.add_alias(training_name + "/avg_best_%i_cpkt/avrg_job" % num_checkpoints)
            training_name = training_name + "/avg_best_%i_cpkt" % num_checkpoints
        else:
            # we only have one
            checkpoint = checkpoints[0]
            training_name = training_name + "/best_cpkt"
    elif get_last_averaged_checkpoint is not None:
        assert get_last_averaged_checkpoint >= 2, "For the single last checkpoint use get_specific_checkpoint instead"
        num_checkpoints = len(train_job.out_checkpoints)
        avg = AverageTorchCheckpointsJob(
            checkpoints=[train_job.out_checkpoints[num_checkpoints - i] for i in range(get_last_averaged_checkpoint)],
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        avg.add_alias(training_name + "/avg_last_%i_cpkt/avrg_job" % num_checkpoints)
        checkpoint = avg.out_checkpoint
        training_name = training_name + "/avg_last_%i_cpkt" % num_checkpoints
    else:
        checkpoint = train_job.out_checkpoints[get_specific_checkpoint]
        training_name = training_name + "/ep_%i_cpkt" % get_specific_checkpoint

    prior_file = None
    if with_prior:
        returnn_config = get_prior_config(
            training_datasets=datasets,
            network_module=train_args["network_module"],
            config=prior_config if prior_config is not None else {},
            net_args=train_args["net_args"],
            unhashed_net_args=train_args.get("unhashed_net_args", None),
            debug=train_args.get("debug", False),
        )
        if "hubert_tune_v1_large" in training_name or "hubert_tune_v2_large" in training_name:
            returnn_config.config["max_seqs"] = 20
        elif "hubert_tune_v1_xlarge" in training_name or "hubert_tune_v2_xlarge" in training_name:
            returnn_config.config["max_seqs"] = 15
        prior_file = compute_prior(
            training_name,
            returnn_config,
            checkpoint=checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(training_name + "/prior.txt", prior_file)
    else:
        if prior_config is not None:
            raise ValueError("prior_config can only be set if with_prior is True")

    asr_model = ASRModel(
        checkpoint=checkpoint,
        network_module=train_args["network_module"],
        net_args=train_args["net_args"],
        prior_file=prior_file,
        prefix_name=training_name,
    )

    return asr_model


def generate_kd_hypothesis(
    prefix_name: str,
    train_job: ReturnnTrainingJob,
    train_args,
    train_data: TrainingDatasets,
    checkpoint: Union[int, str],
    decoder_config,
    prior_scale: float,
    lm_scale: float,
    train_referece: Optional[tk.Path] = None,
    decoder: str = "ctc.decoder.flashlight_ctc_kdhyps",
    debug=False,
):
    decoder_config = copy.deepcopy(decoder_config)
    decoder_config.lm_weight = lm_scale
    decoder_config.prior_scale = prior_scale
    if checkpoint == "best4":
        asr_model = prepare_asr_model(
            prefix_name,
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
        )
    else:
        asr_model = prepare_asr_model(
            prefix_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=checkpoint
        )
    decoder_args = {"config": asdict(decoder_config), "extra_config": {"print_rtf": False, "print_hypothesis": False}}
    decoder_args["config"]["prior_file"] = asr_model.prior_file
    returnn_search_config = get_forward_config(
        network_module=asr_model.network_module,
        config={},
        net_args=asr_model.net_args,
        decoder_args=decoder_args,
        decoder=decoder,
        debug=debug,
    )
    returnn_config = copy.deepcopy(returnn_search_config)
    returnn_config.config["forward"] = copy.deepcopy(train_data.train.as_returnn_opts())
    del returnn_config.config["forward"]["datasets"]["zip_dataset"]["partition_epoch"]
    returnn_config.config["forward"]["datasets"]["zip_dataset"]["seq_ordering"] = "sorted"
    returnn_config.config["batch_size"] = 250 * 16000
    search_job = ReturnnForwardJobV2(
        model_checkpoint=asr_model.checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=10,
        time_rqmt=168,
        device="gpu",
        cpu_rqmt=2,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=["search_out.py", "n_best_probs.py"],
    )
    prefix_name = prefix_name + "/search_lm%.1f_prior%.1f" % (lm_scale, prior_scale)
    search_job.add_alias(prefix_name + "/n_best_probs_job")

    if train_referece is not None:
        search_ctm_job = SearchWordsToCTMJob(
            recog_words_file=search_job.out_files["search_out.py"],
            bliss_corpus=train_referece,
        )
        search_ctm_job.add_alias(prefix_name + "/ctm_job")
        search_ctm = search_ctm_job.out_ctm_file

        stm_file = CorpusToStmJob(bliss_corpus=train_referece).out_stm_path

        sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=3)
        sclite_job.add_alias(prefix_name + "/sclite_job")
        tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
        tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return search_job.out_files["n_best_probs.py"], asr_model.prior_file, asr_model.checkpoint


def calculate_blank_counts(
    prefix_name: str,
    train_job: ReturnnTrainingJob,
    train_args,
    train_data: TrainingDatasets,
    checkpoint: Union[int, str],
    debug=False,
):
    if checkpoint == "best4":
        asr_model = prepare_asr_model(
            prefix_name,
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
        )
    else:
        asr_model = prepare_asr_model(
            prefix_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=checkpoint
        )
    post_config = {
        "num_workers_per_gpu": 2,
    }

    base_config = {
        #############
        "batch_size": 500 * 16000,
        "max_seqs": 240,
        #############
        "forward": copy.deepcopy(train_data.cv.as_returnn_opts()),
    }
    config = {**base_config, **copy.deepcopy({})}
    post_config["backend"] = "torch"

    serializer = serialize_forward(
        network_module=train_args["network_module"],
        net_args=train_args["net_args"],
        unhashed_net_args=train_args.get("unhashed_net_args", None),
        forward_module=None,  # same as network
        forward_step_name="calc_blank",
        forward_init_args=None,
        unhashed_forward_init_args=None,
        debug=debug,
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    search_job = ReturnnForwardJobV2(
        model_checkpoint=asr_model.checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=10,
        time_rqmt=2,
        device="gpu",
        cpu_rqmt=8,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        output_files=["blank_counts.npy"],
    )
    search_job.add_alias(prefix_name + "/calculate_blank_counts")
    return search_job.out_files["blank_counts.npy"]
