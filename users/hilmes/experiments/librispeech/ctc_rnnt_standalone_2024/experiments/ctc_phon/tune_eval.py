from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import search, ASRModel, prepare_asr_model
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
from typing import List, Optional, Dict, Any, List, Union
from ...data.common import TrainingDatasets
from dataclasses import dataclass, asdict
import copy
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from sisyphus import tk
from i6_core.returnn.training import ReturnnTrainingJob

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": MINI_RETURNN_ROOT,
}


def eval_model(
    training_name: str,
    train_job: ReturnnTrainingJob,
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    decoder_config: DecoderConfig,
    dev_dataset_tuples: Dict[str, Any],
    result_dict: Optional[Dict[str, Any]] = None,
    lm_scales: Optional[List[float]] = None,
    prior_scales: Optional[List[float]] = None,
    specific_epoch: Optional[Union[int, List]] = None,
    decoder_module: str = "ctc.decoder.flashlight_ctc_v1",
    loss_name: str = "dev_loss_ctc",
):
    if specific_epoch is None:
        specific_epoch = train_job.returnn_config.post_config["num_epochs"]
    if isinstance(specific_epoch, int):
        specific_epoch = [specific_epoch]
    if lm_scales is None:
        lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    if prior_scales is None:
        prior_scales = [0.7, 0.9]
    if result_dict is None:
        result_dict = {}
    for epoch in specific_epoch:
        asr_model = prepare_asr_model(
            training_name + f"/{epoch}",
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_specific_checkpoint=epoch,
        )
        res = tune_and_evaluate_helper(
            training_name + f"/{epoch}",
            asr_model,
            decoder_config,
            lm_scales=lm_scales,
            prior_scales=prior_scales,
            dev_dataset_tuples=dev_dataset_tuples,
            decoder_module=decoder_module,
        )
        result_dict.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(4, loss_name),
    )
    res = tune_and_evaluate_helper(
        training_name + "/best4",
        asr_model_best4,
        decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
        decoder_module=decoder_module,
    )
    result_dict.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(1, loss_name),
    )
    res = tune_and_evaluate_helper(
        training_name + "/best",
        asr_model_best,
        decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
        decoder_module=decoder_module,
    )
    result_dict.update(res)
    return result_dict


def tune_and_evaluate_helper(
    training_name: str,
    asr_model: ASRModel,
    base_decoder_config: DecoderConfig,
    lm_scales: List[float],
    prior_scales: List[float],
    dev_dataset_tuples: Dict[str, Any],
    quant_str: Optional[str] = None,
    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    decoder_module: str = "ctc.decoder.flashlight_ctc_v1",
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
    tune_values_clean = []
    tune_values_other = []
    results = {}
    for lm_weight in lm_scales:
        for prior_scale in prior_scales:
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = lm_weight
            decoder_config.prior_scale = prior_scale
            search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
            search_jobs, wers = search(
                search_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                **default_returnn,
            )
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_clean.append((wers[search_name + "/dev-clean"]))
            tune_values_other.append((wers[search_name + "/dev-other"]))
            results.update(wers)
    if test_dataset_tuples is not None and False:
        # TODO
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn,
            )
            report_values[key] = wers[training_name + "/" + key]
    return results


from i6_core.util import instanciate_delayed


def build_base_report(report: Dict):
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        new_dic = {k: v for k, v in dic.items() if "other" in k}
        if all(new_dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(new_dic[best])), best)
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
        new_dic = {k: v for k, v in dic.items() if "other" in k}
        if all(dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(new_dic[best])), best)
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
    ]
    line.append("Baselines")
    for exp, value in best_dc.items():
        if (
            not any(name in exp.split("_")[-1] for name in exps + ["True", "False"])
            and not any(exp.endswith(name) for name in exps + ["True", "False"])
            and not ["elim", "blank"] == exp.split("_")[-3:-1]
            and not "trim_blanks" in exp
        ):
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
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
    best_dc = tmp
    assert len(best_dc) == 0, best_dc
    return "\n".join(line)
