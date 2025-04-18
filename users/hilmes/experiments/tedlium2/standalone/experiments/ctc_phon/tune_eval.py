import os.path

from ...default_tools import RETURNN_EXE, QUANT_RETURNN, MINI_RETURNN_ROOT
from ...pipeline import search, ASRModel, quantize_static, prepare_asr_model
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
from typing import List, Optional, Dict, Any, List, Union, Tuple
from ...data.common import TrainingDatasets
from dataclasses import dataclass, asdict
from ...config import get_static_quant_config
import copy
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob, ReturnnForwardJobV2
from functools import partial


@dataclass
class QuantArgs:
    sample_ls: List[int]
    quant_config_dict: Dict[str, Any]
    decoder: str
    num_iterations: int
    datasets: TrainingDatasets
    network_module: str
    filter_args: Optional[Dict[str, Any]] = None


@dataclass
class RTFArgs:
    beam_sizes: Optional[List[int]] = None
    beam_size_tokens: Optional[List[int]] = None
    beam_thresholds: Optional[List[int]] = None
    decoder_module: Optional[str] = None


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
    import_memristor: bool = False,
    use_gpu: bool = False,
    extra_forward_config: Optional[dict[str, Any]] = None,
    run_best_4: bool = True,
    run_best: bool = True,
    run_test: bool = False,
    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    prior_args: Optional[Dict[str, Any]] = None,
    run_rtf: bool = False,  # for now only for last epoch
    rtf_args: Optional[RTFArgs] = None,
    with_prior: bool = True,
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
    debug = train_args.get("debug", False)
    for epoch in specific_epoch:
        asr_model = prepare_asr_model(
            training_name + f"/{epoch}",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=with_prior,
            datasets=train_data,
            get_specific_checkpoint=epoch,
            prior_config={"import_memristor": import_memristor} if import_memristor is True else None,
        )
        if prior_args is not None:
            asr_model.net_args = train_args["net_args"]
            asr_model.network_module = train_args["network_module"]
        res, _ = tune_and_evaluate_helper(
            training_name + f"/{epoch}",
            asr_model,
            decoder_config,
            lm_scales=lm_scales,
            prior_scales=prior_scales,
            dev_dataset_tuples=dev_dataset_tuples,
            decoder_module=decoder_module,
            use_gpu=use_gpu,
            import_memristor=import_memristor,
            debug=debug,
            extra_forward_config=extra_forward_config,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            run_rtf=run_rtf,
            rtf_args=rtf_args,
        )
        result_dict.update(res)
    if run_best_4 is True:
        asr_model_best4 = prepare_asr_model(
            training_name + "/best4",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=with_prior,
            datasets=train_data,
            get_best_averaged_checkpoint=(4, loss_name),
            prior_config={"import_memristor": import_memristor} if import_memristor is True else None,
        )
        if prior_args is not None:
            asr_model_best4.net_args = train_args["net_args"]
            asr_model_best4.network_module = train_args["network_module"]
        res, _ = tune_and_evaluate_helper(
            training_name + "/best4",
            asr_model_best4,
            decoder_config,
            lm_scales=lm_scales,
            prior_scales=prior_scales,
            dev_dataset_tuples=dev_dataset_tuples,
            decoder_module=decoder_module,
            use_gpu=use_gpu,
            import_memristor=import_memristor,
            debug=debug,
            extra_forward_config=extra_forward_config,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
        )
        result_dict.update(res)
    if run_best is True:
        asr_model_best = prepare_asr_model(
            training_name + "/best",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=with_prior,
            datasets=train_data,
            get_best_averaged_checkpoint=(1, loss_name),
            prior_config={"import_memristor": import_memristor} if import_memristor is True else None,
        )
        if prior_args is not None:
            asr_model_best.net_args = train_args["net_args"]
            asr_model_best.network_module = train_args["network_module"]
        res, _ = tune_and_evaluate_helper(
            training_name + "/best",
            asr_model_best,
            decoder_config,
            lm_scales=lm_scales,
            prior_scales=prior_scales,
            dev_dataset_tuples=dev_dataset_tuples,
            decoder_module=decoder_module,
            use_gpu=use_gpu,
            import_memristor=import_memristor,
            debug=debug,
            extra_forward_config=extra_forward_config,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
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
    quant_args: Optional[QuantArgs] = None,
    decoder_module: str = "ctc.decoder.flashlight_ctc_v1",
    import_memristor: bool = False,
    extra_forward_config: Optional[dict[str, Any]] = None,
    use_gpu: bool = False,
    debug: bool = False,
    run_test: bool = False,
    run_rtf: bool = False,
    rtf_args: Optional[RTFArgs] = None,
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
            if not lm_weight == 0.0:
                decoder_config.lm_weight = lm_weight
            if not prior_scale == 0.0:
                decoder_config.prior_scale = prior_scale
            #else:
            #    assert asr_model.prior_file is None, "Prior scale is set to 0"
            search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
            search_jobs, wers = search(
                search_name,
                forward_config=extra_forward_config or {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                use_gpu=use_gpu,
                import_memristor=import_memristor,
                debug=debug,
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
                    dataset_filter_args=quant_args.filter_args,
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
                    prefix_name=it_name,
                )
                for lm_weight in lm_scales:
                    for prior_scale in prior_scales:
                        decoder_config = copy.deepcopy(base_decoder_config)
                        decoder_config.lm_weight = lm_weight
                        decoder_config.prior_scale = prior_scale
                        search_name = it_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                        search_jobs, wers = search(
                            search_name,
                            forward_config=extra_forward_config or {},
                            asr_model=quant_model,
                            decoder_module=quant_args.decoder,
                            decoder_args={"config": asdict(decoder_config)},
                            test_dataset_tuples=dev_dataset_tuples,
                            use_gpu=use_gpu,
                            debug=debug,
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
                                use_gpu=use_gpu,
                                debug=debug,
                                **default_returnn,
                            )
                            results.update(wers)
    pick_optimal_params_job = None
    if run_test is True and test_dataset_tuples is not None and False:
        for key, tune_values in [("test", tune_values)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config=extra_forward_config or {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={"test": test_dataset_tuples["test"]},
                use_gpu=use_gpu,
                **default_returnn,
            )
        results.update(wers)
    assert not rtf_args or run_rtf is True
    if run_rtf is True:
        for key, tune_values in [("test", tune_values)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            run_rtf_test(
                search_name=training_name + f"/rtf_amd",
                base_decoder_config=base_decoder_config,
                lm_scales=[pick_optimal_params_job.out_optimal_parameters[0]],
                prior_scales=[pick_optimal_params_job.out_optimal_parameters[1]],
                dev_dataset_tuples=dev_dataset_tuples,
                device="amd",
                asr_model=asr_model,
                rtf_args=rtf_args,
            )

    return results, pick_optimal_params_job


def run_rtf_test(
    search_name: str,
    base_decoder_config: DecoderConfig,
    lm_scales: List[float],
    prior_scales: List[float],
    dev_dataset_tuples: Dict[str, Any],
    asr_model: ASRModel,
    device: str,
    import_memristor: bool = False,
    extra_forward_config: Optional[dict[str, Any]] = None,
    use_gpu: bool = False,
    debug: bool = False,
    rtf_args: Optional[RTFArgs] = None,
):
    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1_rescale_measure import DecoderConfig


    decoder_module = rtf_args.decoder_module or "ctc.decoder.flashlight_ctc_v1_rescale_measure"

    report = {}
    for lm_weight in lm_scales:
        for prior_scale in prior_scales:
            beam_sizes = rtf_args.beam_sizes or [base_decoder_config.beam_size]
            beam_size_tokens = rtf_args.beam_size_tokens or [base_decoder_config.beam_size_token]
            beam_thresholds = rtf_args.beam_thresholds or [base_decoder_config.beam_threshold]
            for beam_size in beam_sizes:
                for beam_size_token in beam_size_tokens:
                    for beam_threshold in beam_thresholds:
                        decoder_config = DecoderConfig(
                            beam_size=beam_size,
                            beam_size_token=beam_size_token,
                            beam_threshold=beam_threshold,
                            lm_weight=lm_weight,
                            prior_scale=prior_scale,
                            lexicon=base_decoder_config.lexicon,
                            returnn_vocab=base_decoder_config.returnn_vocab,
                            energy_device=device,
                            arpa_lm=base_decoder_config.arpa_lm,
                        )
                        name = search_name + f"/{beam_size}_{beam_size_token}_{beam_threshold}"
                        search_jobs: List[ReturnnForwardJobV2]
                        search_jobs, wers = search(
                            name,
                            forward_config=extra_forward_config or {"num_workers_per_gpu": 0},
                            asr_model=asr_model,
                            decoder_module=decoder_module,
                            decoder_args={"config": asdict(decoder_config)},
                            test_dataset_tuples=dev_dataset_tuples,
                            use_gpu=use_gpu,
                            import_memristor=import_memristor,
                            debug=debug,
                            additional_outputs=["rtf", "energy"],
                            **default_returnn,
                        )
                        for job in search_jobs:
                            job.rqmt["sbatch_args"] = f"-p rescale_{device} -A rescale_speed"
                            job.rqmt["cpu"] = 2
                        assert len(search_jobs) == 1, "Only one search job is supported for now"
                        tk.register_output(
                            search_name + f"/rtf_{beam_size}_{beam_size_token}_{beam_threshold}",
                            search_jobs[0].out_files["rtf"],
                        )
                        tk.register_output(
                            search_name + f"/energy_{beam_size}_{beam_size_token}_{beam_threshold}",
                            search_jobs[0].out_files["energy"],
                        )
                        tk.register_output(
                            search_name + f"/wer_{beam_size}_{beam_size_token}_{beam_threshold}", list(wers.values())[0]
                        )
                        report[f"{beam_size}_{beam_size_token}_{beam_threshold}"] = (
                            search_jobs[0].out_files["rtf"],
                            search_jobs[0].out_files["energy"],
                            list(wers.values())[0],
                        )
    tk.register_report(
        f"reports/{search_name.split('/')[7]}/{search_name.split('/')[5]}", partial(build_rtf_report, report)
    )


from i6_core.util import instanciate_delayed


def build_report(report: Dict):
    line = []
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[" ".join(exp.split("/")[4:6])] = dic[best]
        else:
            best_dc[" ".join(exp.split("/")[4:6])] = "None"
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
            layer, size = exp.split("_")[6:8]
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
            best_baselines[" ".join(exp.split("/")[4:])] = "{:.1f}".format(float(dic[best]))
        else:
            best_baselines[" ".join(exp.split("/")[4:])] = "None"
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[" ".join(exp.split("/")[4:])] = "{:.1f}".format(float(dic[best]))
        else:
            best_dc[" ".join(exp.split("/")[4:])] = "None"
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
                line.append(
                    f"Dim {size} Layer {layer} Baseline {best_baselines[f'ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_{layer}_{size}_sub4_50eps']}"
                )
            elif any("pos_enc_w_hubert" in exp for exp in best_dc):
                line.append(
                    f"Dim {size} Layer {layer} Baseline {best_baselines[f'ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_500_{size}_{4}_8_{11}']}"
                )
            else:
                line.append(
                    f"Dim {size} Layer {layer} Baseline {best_baselines[f'distill_auxloss ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1_{layer}_{size}']}"
                )
            nl = [""]
            for t in sorted(ts):
                nl.append(f"{t}".rjust(7))
            line.append(" ".join(nl))
            for scale in sorted(scales):
                nl = ["{:.2f}".format(float(scale)).ljust(6)]
                for t in sorted(ts):
                    if pref + "_" + str(layer) + "_" + str(size) + "_" + str(scale) + "_" + str(t) in best_dc:
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
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(dic[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test" in dic:
                best_dc["/".join(best.split("/")[:-2]) + "/test"] = (
                    "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test"])),
                    best,
                )
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    return "\n".join(line)


def build_base_report(report: Dict):
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        tmp = {x: dic[x] for x in dic.keys() if not "test" in x}
        if all(tmp.values()):
            best = min(tmp, key=tmp.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(tmp[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test" in dic:
                if dic["/".join(best.split("/")[:-2]) + "/test"] is not None:
                    best_dc["/".join(best.split("/")[:-2]) + "/test"] = (
                        "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test"])),
                        best,
                    )
                else:
                    best_dc["/".join(best.split("/")[:-2]) + "/test"] = ("None", "")
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    return "\n".join(line)


def build_rtf_report(report: Dict):
    beam_sizes = set()
    beam_size_tokens = set()
    beam_thresholds = set()

    report = copy.deepcopy(report)
    instanciate_delayed(report)
    for exp in report:
        beam, token, thresh = exp.split("_")
        beam_sizes.add(int(beam))
        beam_size_tokens.add(int(token))
        beam_thresholds.add(int(thresh))

    line = []
    min_line = ""
    min_score = 10000
    line.append(
        "Beam".ljust(7)
        + f"Token".ljust(7)
        + "Thresh".ljust(7)
        + "WER".ljust(7)
        + "Search RTF".ljust(12)
        + "Energy".ljust(12)
        + "AM RTF".ljust(7)
        + "Total".ljust(7)
    )
    for beam in sorted(beam_sizes):
        for token in sorted(beam_size_tokens):
            for thresh in sorted(beam_thresholds):
                if os.path.exists(report[f"{beam}_{token}_{thresh}"][0]):
                    rtf = open(report[f"{beam}_{token}_{thresh}"][0], "rt").read()
                    search_rtf = rtf.split(",")[2].split(":")[1].split("\n")[0].strip()
                    am_rtf = rtf.split(",")[1].split(":")[1].split("\n")[0].strip()
                    total_rtf = rtf.split(",")[4].split(":")[1].split("\n")[0].strip()
                else:
                    search_rtf = "None"
                    am_rtf = "None"
                    total_rtf = "None"

                if os.path.exists(report[f"{beam}_{token}_{thresh}"][1]):
                    energy = float(open(report[f"{beam}_{token}_{thresh}"][1], "rt").read()) / 3600
                else:
                    energy = "0"
                wer = report[f"{beam}_{token}_{thresh}"][2] or "0"

                line.append(
                    f"{beam}".ljust(7)
                    + f"{token}".ljust(7)
                    + f"{thresh}".ljust(7)
                    + f"{wer}".ljust(7)
                    + f"{search_rtf} ".ljust(12)
                    + f"{float(energy):.2f}".ljust(12)
                    + f"{am_rtf} ".ljust(7)
                    + f"{total_rtf} ".ljust(7)
                    + f"{float(wer):.1f}".ljust(7)
                    + f"{float(wer):.2f}".ljust(7)
                )
                if isinstance(wer, float) and wer < min_score:
                    min_line = (
                        f"{beam}".ljust(7)
                        + f"{token}".ljust(7)
                        + f"{thresh}".ljust(7)
                        + f"{wer}".ljust(7)
                        + f"{search_rtf} ".ljust(12)
                        + f"{float(energy):.2f}".ljust(12)
                        + f"{am_rtf} ".ljust(7)
                        + f"{total_rtf} ".ljust(7)
                        + f"{float(wer):.1f}".ljust(7)
                        + f"{float(wer):.2f}".ljust(7)
                    )
                    min_score = wer

    line.insert(0, min_line)
    line.insert(1, " ")
    return "\n".join(line)


def build_hubert_distill_report(report: Dict):
    report = copy.deepcopy(report)
    baselines = report.pop("baselines", {})
    best_baselines = {}
    for exp, dic in baselines.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_baselines[" ".join(exp.split("/")[4:])] = (dic[best], best)
        else:
            best_baselines[" ".join(exp.split("/")[4:])] = ("None", "")
    best_dc = {}
    for exp, best in best_baselines.items():
        best_dc[exp] = best
    for exp, dic in report.items():
        instanciate_delayed(dic)
        tmp = {x: dic[x] for x in dic.keys() if not "test" in x}
        if all(tmp.values()):
            best = min(tmp, key=tmp.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(tmp[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test" in dic:
                if dic["/".join(best.split("/")[:-2]) + "/test"] is not None:
                    best_dc["/".join(best.split("/")[:-2]) + "/test"] = (
                        "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test"])),
                        best,
                    )
                else:
                    best_dc["/".join(best.split("/")[:-2]) + "/test"] = ("None", "")
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
        "mix",
        "pretrain",
        "elim_blank_prior",
        "kdhyps",
        "trim_blanks",
        "sym",
        "rdn",
        "thresh",
    ]
    line.append("Baselines")
    tmp = copy.deepcopy(best_dc)
    for exp, value in best_dc.items():
        if (
            not any(name in exp.split("_")[-1] or exp.endswith(name) for name in exps + ["True", "False"])
            and not ["elim", "blank"] == exp.split("_")[-3:-1]
            and not "trim_blanks" in exp
            and not "test" in exp
        ):
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
            del tmp[exp]
            if "/".join(value[1].split("/")[:-2]) + "/test" in best_dc:
                value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test"]
                line.append(
                    f"{' '.join(value_test[1].split('.')[2:-2])+ '/test'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                )
                del tmp["/".join(value[1].split("/")[:-2]) + "/test"]
    best_dc = tmp
    # best_dc = {
    #     exp: value
    #     for exp, value in best_dc.items()
    #     if (
    #         any(exp.endswith(name) or name in exp.split("_")[-1] for name in exps + ["True", "False"])
    #         or ["elim", "blank"] == exp.split("_")[-3:-1]
    #         or "trim_blanks" in exp
    #     )
    # }
    line.append("")
    tmp = copy.deepcopy(best_dc)
    for name in exps:
        best_dc = copy.deepcopy(tmp)
        line.append(name)
        for exp, value in best_dc.items():
            if "test" in exp:
                continue
            if exp.endswith(name):
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
                if "/".join(value[1].split("/")[:-2]) + "/test" in best_dc:
                    value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test"]
                    line.append(
                        f"{' '.join(value_test[1].split('.')[2:-2]) + '/test'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                    )
                    del tmp["/".join(value[1].split("/")[:-2]) + "/test"]
            elif name == "keepsome" and "keepsome" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "mix" and "mix" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif (
                name == "elim_blank num"
                and ["elim", "blank"] == exp.split("_")[-3:-1]
                and exp.split("_")[-1].isnumeric()
            ):
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "trim_blanks" and "trim_blanks" in exp:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "keep" and "keep" == exp.split("_")[-2]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "increase" and "increase" in exp.split("_")[-2]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
            elif name == "rdn" and "rdn" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
                if "/".join(value[1].split("/")[:-2]) + "/test" in best_dc:
                    value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test"]
                    line.append(
                        f"{' '.join(value_test[1].split('.')[2:-2]) + '/test'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                    )
                    del tmp["/".join(value[1].split("/")[:-2]) + "/test"]

            elif name == "thresh" and "thresh" in exp.split("_")[-1]:
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
                if "/".join(value[1].split("/")[:-2]) + "/test" in best_dc:
                    value_test = best_dc["/".join(value[1].split("/")[:-2]) + "/test"]
                    line.append(
                        f"{' '.join(value_test[1].split('.')[2:-2]) + '/test'}: {value_test[0]}   {' '.join(value_test[1].split('/')[6:])}"
                    )
                    del tmp["/".join(value[1].split("/")[:-2]) + "/test"]

        line.append("")

    best_dc = copy.deepcopy(tmp)
    line.append("Testsets")
    for exp, value in best_dc.items():
        if "test" in exp:
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
            del tmp[exp]
    best_dc = tmp
    assert len(best_dc) == 0, best_dc
    return "\n".join(line)


def build_qat_report(report: Dict):
    import numpy as np

    exps = ["cycle", "smaller", "greedy"]

    best_dc = {}
    bits = [8, 7, 6, 5, 4, 3, 2, 1.5]
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            if "cycle" in exp:
                mean = np.mean(list(dic.values()))
                mini = np.min(list(dic.values()))
                maxi = np.max(list(dic.values()))
                std = np.std(list(dic.values()))
                best_dc[" ".join(exp.split("/")[5:])] = (
                    "{:.1f}".format(float(dic[best])),
                    best + f"  {mean=} {std=} {mini=} {maxi=}",
                )
            else:
                best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(dic[best])), best)
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    tmp = copy.deepcopy(best_dc)
    for bit in bits:
        best_dc = tmp
        tmp = copy.deepcopy(best_dc)
        first = False
        for exp, value in best_dc.items():
            if any(x in exp for x in exps):
                continue
            if f"{bit}_8" not in exp:
                continue
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
            if first == False:
                first = True
            del tmp[exp]
        if first == True:
            line.append("")
    tmp = best_dc
    for x in exps:
        first = True
        best_dc = copy.deepcopy(tmp)
        for exp, value in best_dc.items():
            if x in exp:
                if first is True:
                    line.append(x)
                    line.append("")
                    first = False
                line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
                del tmp[exp]
        if first is False:
            line.append("")
    assert len(tmp) == 0, tmp
    return "\n".join(line)
