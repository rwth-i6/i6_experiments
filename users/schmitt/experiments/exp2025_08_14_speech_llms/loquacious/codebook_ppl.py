import copy
import enum
from dataclasses import dataclass, asdict
import os.path
from typing import Any, Dict, List, Optional, Tuple, Union

from sisyphus import tk

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.recognition.scoring import ScliteJob

from i6_core.returnn.search import SearchOutputRawReplaceJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.returnn.training import ReturnnTrainingJob, AverageTorchCheckpointsJob, GetBestPtCheckpointJob
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.setups.returnn.datasets import Dataset

from .config import get_forward_config, get_training_config, get_prior_config
from .data.common import TrainingDatasets, get_devtest_all_stms
from ..default_tools import SCTK_BINARY_PATH, RETURNN_EXE, RETURNN_ROOT

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def calculate_wav2vec_codebook_ppl(
    prefix_name: str,
    data: Dataset,
    wav2vec_checkpoint: tk.Path,
    wav2vec_config: tk.Path,
):
    returnn_search_config = get_forward_config(
        network_module=asr_model.network_module,
        config=forward_config,
        net_args=asr_model.net_args,
        decoder_args=decoder_args,
        decoder=decoder_module,
        debug=debug,
        vocab_opts=vocab_opts,
    )
    tune_and_evaluate_helper(
        training_name + f"/{epoch}",
        asr_model,
        decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
        decoder_module=decoder_module,
        use_gpu=use_gpu,
        debug=debug,
        extra_forward_config=extra_forward_config,
        run_test=run_test,
        test_dataset_tuples=test_dataset_tuples,
        vocab_opts=train_data.train.dataset.target_options
    )


def tune_and_evaluate_helper(
    training_name: str,
    asr_model: ASRModel,
    base_decoder_config: DecoderConfig,
    lm_scales: List[float],
    prior_scales: List[float],
    dev_dataset_tuples: Dict[str, Any],
    vocab_opts: Dict,
    quant_str: Optional[str] = None,
    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    quant_args: Optional[Any] = None,
    decoder_module: str = "ctc.decoder.flashlight_ctc_v1",
    extra_forward_config: Optional[dict[str, Any]] = None,
    use_gpu: bool = False,
    debug: bool = False,
    run_test: bool = False,
):
    """
    Example helper to execute tuning over lm_scales and prior scales.
    With the best values runs test-clean and test-other.

    This is just a reference helper and can (should) be freely changed, copied, modified etc...

    :param training_name: for alias and output names
    :param asr_model: ASR model to use
    :param base_decoder_config: any decoder config dataclass
    :param lm_scales: lm scales for tuning
    :param prior_scales: prior scales for tuning, same length as lm scales
    """
    tune_parameters = []
    tune_values_dev = []
    report_values = {}
    for lm_weight in lm_scales:
        for prior_scale in prior_scales:
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = lm_weight
            decoder_config.prior_scale = prior_scale
            search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
            search_jobs, wers, ctms = search(
                search_name,
                forward_config=extra_forward_config or {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                use_gpu=use_gpu,
                debug=debug,
                vocab_opts=vocab_opts,
                **default_returnn,
            )
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_dev.append((wers[search_name + "/dev.short"]))
            # report_values.update(wers)

    """
    IMPORTANT! We treat all dev and test datasets as "test" datasets, as the LM tuning is done on dev.all.short
    """

    all_ctms = {
        "dev": {},
        "test": {},
    }
    for type in ["dev", "test"]:
        test_tuples = [
            (f"{type}.commonvoice", tune_values_dev),
            (f"{type}.librispeech", tune_values_dev),
            (f"{type}.voxpopuli", tune_values_dev),
            (f"{type}.yodas", tune_values_dev),
        ]
        for key, tune_values in test_tuples:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"

            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers, ctms = search(
                training_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
                vocab_opts=vocab_opts,
                debug=debug,
                **default_returnn,
            )

            all_ctms[type].update(ctms)
            # report_values[key] = wers[key]
    evaluate_all(prefix_name=training_name, dev_ctms=all_ctms["dev"], test_ctms=all_ctms["test"])


from i6_core.util import instanciate_delayed


def build_base_report(report: Dict):
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        new_dic = {k: v for k, v in dic.items() if "other" in k}
        if all(new_dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(new_dic[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test-other" in dic:
                best_dc["/".join(best.split("/")[:-2]) + "/test-other"] = (
                    "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test-other"])),
                    best,
                )
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    return "\n".join(line)


def build_hubert_distill_report(report: Dict):

    report = copy.deepcopy(report)
    baselines = report.pop("baselines")
    best_baselines = {}
    for exp, dic in baselines.items():
        instanciate_delayed(dic)
        new_dic = {k: v for k, v in dic.items() if "other" in k}
        if all(dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_baselines[" ".join(exp.split("/")[4:])] = (new_dic[best], best)
        else:
            best_baselines[" ".join(exp.split("/")[4:])] = ("None", "")
    best_dc = {}
    for exp, best in best_baselines.items():
        best_dc[exp] = best
    for exp, dic in report.items():
        instanciate_delayed(dic)
        new_dic = {k: v for k, v in dic.items() if "other" in k and "test" not in k}
        if all(new_dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(new_dic[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test-other" in dic:
                if dic["/".join(best.split("/")[:-2]) + "/test-other"] is not None:
                    best_dc["/".join(best.split("/")[:-2]) + "/test-other"] = (
                        "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test-other"])),
                        best,
                    )
                else:
                    best_dc["/".join(best.split("/")[:-2]) + "/test-other"] = ("None", "")
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    # line.append("Small")
    # for exp, value in best_dc.items():
    #     if "128" in exp:
    #         line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    # best_dc = {exp: value for exp, value in best_dc.items() if "128" not in exp}
    # line.append("")

    exps = [
        "elim_blank",
        "keepsome",
        "sym",
        "mix",
        # "pretrain",
        "elim_blank_prior",
        "kdhyps",
        "trim_blanks",
        "elim_blank num",
        # "long",
        "random",
        "thresh",
    ]
    line.append("Baselines")
    for exp, value in best_dc.items():
        if (
            not any(name in exp.split("_")[-1] for name in exps)
            and not any(exp.endswith(name) for name in exps + ["True", "False"])
            and not ["elim", "blank"] == exp.split("_")[-3:-1]
            and not "trim_blanks" in exp
            and not "test" in exp
        ):
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
            if "/".join(value[1].split("/")[:-2]) + "/test-other" in best_dc:
                value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test-other"]
                line.append(
                    f"{' '.join(value[1].split('.')[2:-2])+ '/test-other'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                )
    best_dc = {
        exp: value
        for exp, value in best_dc.items()
        if (
            any(exp.endswith(name) for name in exps + ["True", "False"])
            or any(name in exp.split("_")[-1] for name in exps + ["True", "False"])
            or ["elim", "blank"] == exp.split("_")[-3:-1]
            or "trim_blanks" in exp
        )
    }
    tmp = copy.deepcopy(best_dc)
    line.append("")
    for name in exps:
        line.append(name)
        for exp, value in best_dc.items():
            if "test" in exp:
                continue
            if exp.endswith(name):
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "keepsome" and "keepsome" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "mix" and "mix" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "elim_blank num" and ["elim", "blank"] == exp.split("_")[-3:-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "trim_blanks" and "trim_blanks" in exp:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "random" and "random" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
                if "/".join(value[1].split("/")[:-2]) + "/test-other" in best_dc:
                    value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test-other"]
                    line.append(
                        f"{' '.join(value[1].split('.')[2:-2]) + '/test-other'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                    )
                    del tmp["/".join(value[1].split("/")[:-2]) + "/test-other"]
            elif name == "thresh" and "thresh" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
                if "/".join(value[1].split("/")[:-2]) + "/test-other" in best_dc:
                    value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test-other"]
                    line.append(
                        f"{' '.join(value[1].split('.')[2:-2]) + '/test-other'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                    )
                    del tmp["/".join(value[1].split("/")[:-2]) + "/test-other"]
        line.append("")
        # best_dc = {
        #     exp: value
        #     for exp, value in best_dc.items()
        #     if not exp.endswith(name)
        #     and not (name == "keepsome" and "keepsome" in exp.split("_")[-1])
        #     and not (name == "mix" and "mix" in exp.split("_")[-1])
        #     and not (name == "elim_blank num" and ["elim", "blank"] == exp.split("_")[-3:-1])
        #     and not (name == "trim_blanks" and "trim_blanks" in exp)
        # }
    # line.append("Warmup")
    # for exp, value in best_dc.items():
    #     if exp.endswith("True") or exp.endswith("False"):
    #         line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    # line.append("")
    # best_dc = {
    #     exp: value for exp, value in best_dc.items() if not any(exp.endswith(name) for name in ["True", "False"])
    # }
    best_dc = copy.deepcopy(tmp)
    line.append("Testsets")
    for exp, value in best_dc.items():
        if "test" in exp:
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
            del tmp[exp]
    best_dc = tmp
    assert len(best_dc) == 0, best_dc
    return "\n".join(line)


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
    mem_rqmt: float = 14,
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
    returnn_config.config["forward_data"] = recognition_dataset.as_returnn_opts()
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
        output_files=["search_out.py.gz"],
    )
    search_job.add_alias(prefix_name + "/search_job")

    words = SearchOutputRawReplaceJob(
      search_job.out_files["search_out.py.gz"],
      [("@@ ", "")],
      output_gzip=True
    ).out_search_results

    search_ctm = SearchWordsToCTMJob(
        recog_words_file=words,
        bliss_corpus=recognition_bliss_corpus,
    ).out_ctm_file

    stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

    sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
    tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
    tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    return sclite_job.out_wer, search_job, search_ctm


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
    vocab_opts: Dict,
    use_gpu: bool = False,
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
        vocab_opts=vocab_opts,
    )

    # use fixed last checkpoint for now, needs more fine-grained selection / average etc. here
    wers = {}
    search_jobs = []
    ctms = {}
    for key, (test_dataset, test_dataset_reference) in test_dataset_tuples.items():
        search_name = prefix_name + "/%s" % key
        if "hubert_tune" in search_name:
            mem = 30
        elif "RelPosEnc" in search_name:
            mem = 16
        else:
            mem = 12
        wers[search_name], search_job, ctms[key] = search_single(
            search_name,
            returnn_search_config,
            asr_model.checkpoint,
            test_dataset,
            test_dataset_reference,
            returnn_exe,
            returnn_root,
            use_gpu=use_gpu,
            mem_rqmt=mem,
        )
        search_jobs.append(search_job)

    return search_jobs, wers, ctms


def evaluate_all(prefix_name: str, dev_ctms: Dict[str, tk.Path], test_ctms: dict[str, tk.Path]):
    """
    Compute the full Loquacious WER based on the given subset ctm dicts
    """
    dev_stm, test_stm = get_devtest_all_stms()

    from i6_core.text.processing import PipelineJob
    dev_ctm_all = PipelineJob(list(dev_ctms.values()), [], zip_output=False, mini_task=True).out
    test_ctm_all = PipelineJob(list(test_ctms.values()), [], zip_output=False, mini_task=True).out

    dev_sclite_job = ScliteJob(ref=dev_stm, hyp=dev_ctm_all, sort_files=True, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
    tk.register_output(prefix_name + "/dev.all/sclite/wer", dev_sclite_job.out_wer)
    tk.register_output(prefix_name + "/dev.all/sclite/report", dev_sclite_job.out_report_dir)

    test_sclite_job = ScliteJob(ref=test_stm, hyp=test_ctm_all, sort_files=True, sctk_binary_path=SCTK_BINARY_PATH, precision_ndigit=2)
    tk.register_output(prefix_name + "/test.all/sclite/wer", test_sclite_job.out_wer)
    tk.register_output(prefix_name + "/test.all/sclite/report", test_sclite_job.out_report_dir)


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
    if "hubert_tune_v2" in prefix_name:
        search_job.rqmt["time"] += 12
        search_job.rqmt["gpu_mem"] = 24
    search_job.add_alias(prefix_name + "/prior_job")
    return search_job.out_files["prior.txt"]


def training(training_name, datasets, train_args, num_epochs, returnn_exe, returnn_root, gpu_mem: int = 11):
    """
    :param training_name:
    :param datasets:
    :param train_args:
    :param num_epochs:
    :param returnn_exe: The python executable to run the job with (when using container just "python3")
    :param returnn_root: Path to a checked out RETURNN repository
    """
    default_rqmt = {
        "mem_rqmt": 24,
        "time_rqmt": 168,
        "cpu_rqmt": 6,
        "log_verbosity": 5,
        "returnn_python_exe": returnn_exe,
        "returnn_root": returnn_root,
    }

    num_gpus = train_args["config"].pop("__num_gpus", 1)
    if num_gpus > 1:
        train_args["config"].update({
            "torch_distributed": {"param_sync_step": 100, "reduce_type": "param"},
            "use_horovod": True,
        })
        default_rqmt.update({
            "distributed_launch_cmd": "torchrun",
            "horovod_num_processes": num_gpus,
            "mem_rqmt": 24
        })

    returnn_config = get_training_config(training_datasets=datasets, **train_args)

    train_job = ReturnnTrainingJob(returnn_config=returnn_config, num_epochs=num_epochs, **default_rqmt)
    if gpu_mem != 11:
        train_job.rqmt["gpu_mem"] = gpu_mem

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
                checkpoints=checkpoints, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT
            )
            avg.rqmt["mem"] = 8
            checkpoint = avg.out_checkpoint
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
            returnn_root=RETURNN_ROOT,
        )
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
        prior_file = compute_prior(
            training_name,
            returnn_config,
            checkpoint=checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
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
