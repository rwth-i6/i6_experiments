from ...default_tools import RETURNN_EXE, QUANT_RETURNN, MINI_RETURNN_ROOT
from ...pipeline import search, ASRModel, quantize_static, prepare_asr_model
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
from typing import List, Optional, Dict, Any, List, Union
from ...data.common import TrainingDatasets
from dataclasses import dataclass, asdict
from ...config import get_static_quant_config
import copy
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from sisyphus import tk
from i6_core.returnn.training import ReturnnTrainingJob
@dataclass
class QuantArgs:
    sample_ls: List[int]
    quant_config_dict: Dict[str, Any]
    decoder: str
    num_iterations: int
    datasets: TrainingDatasets
    network_module: str
    filter_args: Optional[Dict[str, Any]] = None
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
        specific_epoch = train_job.returnn_config.post_config['num_epochs']
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
            training_name+f"/{epoch}", train_job, train_args, with_prior=True, datasets=train_data,
            get_specific_checkpoint=epoch
        )
        res, _ = tune_and_evaluate_helper(
            training_name+f"/{epoch}", asr_model, decoder_config, lm_scales=lm_scales,
            prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples,
            decoder_module=decoder_module
        )
        result_dict.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(4, loss_name)
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, decoder_config,
                                      lm_scales=lm_scales, prior_scales=prior_scales,
                                      dev_dataset_tuples=dev_dataset_tuples, decoder_module=decoder_module)
    result_dict.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(1, loss_name)
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best", asr_model_best, decoder_config,
                                      lm_scales=lm_scales, prior_scales=prior_scales,
                                      dev_dataset_tuples=dev_dataset_tuples, decoder_module=decoder_module)
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
    quant_args: Optional[QuantArgs] = None,
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
    tune_values = []
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
            tune_values.append((wers[search_name + "/dev"]))
            results.update(wers)
    if quant_args is not None:
        assert quant_str is not None, "You want your quant to have a name"
        for num_samples in quant_args.sample_ls:
            for seed in range(quant_args.num_iterations):
                it_name = training_name + quant_str + f"/quantize_static/samples_{num_samples}/seed_{seed}"
                quant_config = get_static_quant_config(
                    training_datasets=quant_args.datasets,
                    network_module=quant_args.network_module,
                    net_args=asr_model.net_args,
                    quant_args=quant_args.quant_config_dict,
                    config={},
                    num_samples=num_samples,
                    dataset_seed=seed,
                    debug=False,
                    dataset_filter_args=quant_args.filter_args
                )
                quant_chkpt = quantize_static(
                    prefix_name=it_name,
                    returnn_config=quant_config,
                    checkpoint=asr_model.checkpoint,
                    returnn_exe=RETURNN_EXE,
                    returnn_root=QUANT_RETURNN,
                )
                quant_model = ASRModel(
                    checkpoint=quant_chkpt,
                    net_args=asr_model.net_args | quant_args.quant_config_dict,
                    network_module=quant_args.network_module,
                    prior_file=asr_model.prior_file,
                    prefix_name=it_name
                )
                for lm_weight in lm_scales:
                    for prior_scale in prior_scales:
                        decoder_config = copy.deepcopy(base_decoder_config)
                        decoder_config.lm_weight = lm_weight
                        decoder_config.prior_scale = prior_scale
                        search_name = it_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                        search_jobs, wers = search(
                            search_name,
                            forward_config={},
                            asr_model=quant_model,
                            decoder_module=quant_args.decoder,
                            decoder_args={"config": asdict(decoder_config)},
                            test_dataset_tuples=dev_dataset_tuples,
                            **default_returnn,
                        )
                        results.update(wers)
                        if test_dataset_tuples is not None and seed in [59, 11, 93, 78, 25, 11, 38, 71, 68, 66, 18, 61]:
                            search_jobs, wers = search(
                                search_name,
                                forward_config={},
                                asr_model=quant_model,
                                decoder_module=quant_args.decoder,
                                decoder_args={"config": asdict(decoder_config)},
                                test_dataset_tuples=test_dataset_tuples,
                                **default_returnn,
                            )
                            results.update(wers)
    pick_optimal_params_job = GetOptimalParametersAsVariableJob(
        parameters=tune_parameters, values=tune_values, mode="minimize"
    )
    pick_optimal_params_job.add_alias(training_name + f"/pick_best_dev")
    if test_dataset_tuples is not None and False:
        for key, tune_values in [("test", tune_values)]:
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
            results.update(wers)
    return results, pick_optimal_params_job


from i6_core.util import instanciate_delayed
def build_report(report: Dict):
    line = []
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[' '.join(exp.split('/')[4:6])] = dic[best]
        else:
            best_dc[' '.join(exp.split('/')[4:6])] = "None"
    sizes = set()
    layers = set()
    for exp in best_dc:
        if exp.endswith("eps"):
            pref = "_".join(exp.split("_")[:-4])
            post = "_" + "_".join(exp.split("_")[-2:])
            layer, size = exp.split("_")[4:6]
        else:
            pref = "_".join(exp.split("_")[:-2])
            post = ""
            layer, size = exp.split("_")[-2:]
        sizes.add(int(size))
        layers.add(int(layer))
    nl = []
    for size in sorted(sizes):
        nl.append(f"{size}".rjust(7))
    line.append(" ".join(nl))
    for layer in sorted(layers):
        nl = [f"{layer}".ljust(3)]
        for size in sorted(sizes):
            nl.append(f"{best_dc[pref + '_' + str(layer) + '_' + str(size) + post]}".ljust(7))
        line.append(" ".join(nl))
    return "\n".join(line)

def build_distill_report(report: Dict):
    report = copy.deepcopy(report)
    baselines = report.pop("baselines")
    best_baselines = {}
    for exp, dic in baselines.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_baselines[' '.join(exp.split('/')[4:])] = dic[best]
        else:
            best_baselines[' '.join(exp.split('/')[4:])] = "None"
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[' '.join(exp.split('/')[4:])] = dic[best]
        else:
            best_dc[' '.join(exp.split('/')[4:])] = "None"
    line = []
    sizes = set()
    ts = set()
    scales = set()
    layers = set()
    for exp in best_dc:
        pref = "_".join(exp.split("_")[:-4])
        layer, size, scale, t = exp.split("_")[-4:]
        sizes.add(int(size))
        ts.add(int(t))
        scales.add(float(scale))
        layers.add(int(layer))
    for layer in sorted(layers):
        for size in sorted(sizes):
            if any(exp.endswith("eps") for exp in best_baselines):
                line.append(f"Dim {size} Layer {layer} Baseline {best_baselines[f'ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_{layer}_{size}_sub4_50eps']}")
            else:
                line.append(f"Dim {size} Layer {layer} Baseline {best_baselines[f'distill_auxloss ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1_{layer}_{size}']}")
            nl = [""]
            for t in sorted(ts):
                nl.append(f"{t}".rjust(7))
            line.append(" ".join(nl))
            for scale in sorted(scales):
                nl = ["{:.2f}".format(float(scale)).ljust(6)]
                for t in sorted(ts):
                    if pref+'_'+str(layer)+'_'+str(size)+'_'+str(scale)+'_'+str(t) in best_dc:
                        nl.append(f"{best_dc[pref+'_'+str(layer)+'_'+str(size)+'_'+str(scale)+'_'+str(t)]}".ljust(7))
                    else:
                        nl.append(f"".ljust(7))
                line.append(" ".join(nl))
            line.append("")
    return "\n".join(line)

def build_hubert_report(report: Dict):
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[' '.join(exp.split('/')[5:])] = (dic[best], best)
        else:
            best_dc[' '.join(exp.split('/')[5:])] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    return "\n".join(line)

def build_qat_report(report: Dict):

    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[' '.join(exp.split('/')[5:])] = dic[best]
        else:
            print(dic.values())
            best_dc[' '.join(exp.split('/')[5:])] = "None"
    line = []
    for exp, value in best_dc.items():
        line.append(f"{' '.join(exp.split('.')[2:])}: {value}")
    return "\n".join(line)
