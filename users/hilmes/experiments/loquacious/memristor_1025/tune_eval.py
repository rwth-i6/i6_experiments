import os.path

from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from .pipeline import search, ASRModel, quantize_static, prepare_asr_model, evaluate_all
from .pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
from .pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
from typing import List, Optional, Dict, Any, List, Union, Tuple
from .data.common import TrainingDatasets
from dataclasses import dataclass, asdict
from .config import get_static_quant_config
import copy
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob, ReturnnForwardJobV2
from functools import partial
from i6_core.rasr.config import WriteRasrConfigJob


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
    type: Optional[str] = None
    include_gpu: bool = False
    include_cpu: bool = True
    run_quant: bool = True
    forward_args: Optional[Dict[str, Any]] = None


@dataclass
class RasrRTFArgs:
    max_beam_size: Optional[List[int]] = None
    score_threshold: Optional[List[float]] = None
    decoder_module: Optional[str] = None
    type: Optional[str] = None
    include_gpu: bool = False
    include_cpu: bool = True
    run_quant: bool = True


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
    lm_scales: Optional[List[Union[float, Tuple[tk.Path, str]]]] = None,
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
    rtf_args: Optional[Union[RTFArgs, RasrRTFArgs]] = None,
    with_prior: bool = True,
    get_best_params: bool = False,
    run_search_on_hpc: bool = False,
    unhashed_decoder_args: Optional = None,
    run_rasr: bool = False,
    split_mem_init: bool = False,
    search_gpu: Optional[int] = None,
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
    best_params = None
    for epoch in specific_epoch:
        asr_model = prepare_asr_model(
            training_name + f"/{epoch}",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=with_prior,
            datasets=train_data,
            get_specific_checkpoint=epoch,
            prior_config={"import_memristor": import_memristor}
            if import_memristor is True else None,
            split_preparation=split_mem_init,
            split_args=train_args if split_mem_init else None,
        )
        if prior_args is not None:
            asr_model.net_args = train_args["net_args"]
            asr_model.network_module = train_args["network_module"]
        if split_mem_init is True:
            asr_model.network_module += "_mem_inited"
        res, best_params = tune_and_evaluate_helper(
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
            get_best_params=get_best_params,
            run_search_on_hpc=run_search_on_hpc,
            unhashed_decoder_args=unhashed_decoder_args,
            run_rasr=run_rasr,
            search_gpu=search_gpu,
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
            prior_config={"import_memristor": import_memristor}
            if import_memristor is True and with_prior is True
            else None,
            split_preparation=split_mem_init,
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
            unhashed_decoder_args=unhashed_decoder_args,
            run_rasr=run_rasr,
            search_gpu=search_gpu,
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
            prior_config={"import_memristor": import_memristor}
            if import_memristor is True and with_prior is True
            else None,
            split_preparation=split_mem_init,
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
            unhashed_decoder_args=unhashed_decoder_args,
            run_rasr=run_rasr,
            search_gpu=search_gpu,
        )
        result_dict.update(res)
    if get_best_params is True:
        return result_dict, best_params
    else:
        return result_dict


def tune_and_evaluate_helper(
    training_name: str,
    asr_model: ASRModel,
    base_decoder_config: Union[DecoderConfig, RasrDecoderConfig],
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
    get_best_params: bool = False,
    run_search_on_hpc: bool = False,
    unhashed_decoder_args: Optional = None,
    run_rasr: bool = False,
    search_gpu: Optional[int] = None,
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
            if isinstance(lm_weight, tuple):
                lm_weight, search_name = lm_weight
            else:
                search_name = "lm%.1f_prior%.1f" % (lm_weight, prior_scale)
            if not lm_weight == 0.0:
                if not run_rasr:
                    decoder_config.lm_weight = lm_weight
                else:
                    decoder_config.rasr_config_file.lib_rasr.lm.scale = lm_weight
            if not prior_scale == 0.0:
                decoder_config.prior_scale = prior_scale
            # else:
            #    assert asr_model.prior_file is None, "Prior scale is set to 0"
            if run_rasr:
                recog_rasr_config_path = WriteRasrConfigJob(
                    decoder_config.rasr_config_file, decoder_config.rasr_post_config
                ).out_config
                decoder_config.rasr_config_file = recog_rasr_config_path
                decoder_config.rasr_post_config = None

            search_name = training_name + f"/search_{search_name}"
            search_jobs, wers = search(
                search_name,
                forward_config=extra_forward_config or {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                unhashed_decoder_args={
                    "extra_config": asdict(unhashed_decoder_args) if unhashed_decoder_args is not None else {}
                },
                test_dataset_tuples=dev_dataset_tuples,
                use_gpu=use_gpu,
                import_memristor=import_memristor,
                debug=debug,
                run_rasr=run_rasr,
                **default_returnn,
            )
            if run_search_on_hpc is True:
                for job in search_jobs:
                    if not os.path.exists(f"{job._sis_path()}/finished.run.1"):  # sync back was successful
                        job.rqmt["cpu"] = 12
                        job.hold()
                        job.move_to_hpc = True
            if search_gpu is not None:
                for job in search_jobs:
                    job.rqmt['gpu_mem'] = search_gpu
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
    if run_test is True and test_dataset_tuples is not None:
        dev_ctm_files = {}
        test_ctm_files = {}
        pick_optimal_params_job = GetOptimalParametersAsVariableJob(
            parameters=tune_parameters, values=tune_values, mode="minimize"
        )
        pick_optimal_params_job.add_alias(training_name + f"/pick_best")
        decoder_config = copy.deepcopy(base_decoder_config)
        if not run_rasr:
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
        else:
            decoder_config.rasr_config_file.lib_rasr.lm.scale = pick_optimal_params_job.out_optimal_parameters[0]
        decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
        if run_rasr:
            recog_rasr_config_path = WriteRasrConfigJob(
                decoder_config.rasr_config_file, decoder_config.rasr_post_config
            ).out_config
            decoder_config.rasr_config_file = recog_rasr_config_path
            decoder_config.rasr_post_config = None
        for key in test_dataset_tuples:
            search_jobs, wers, ctm_file = search(
                training_name,
                forward_config=extra_forward_config or {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
                run_rasr=run_rasr,
                return_ctm=True,
                import_memristor=import_memristor,
                **default_returnn,
            )
            assert len(search_jobs) == 1, "In testing only one parameter setting is done"
            if "test" in key:
                test_ctm_files[key] = ctm_file
            else:
                dev_ctm_files[key] = ctm_file
            results.update(wers)
        dev_wer, test_wer = evaluate_all(prefix_name=training_name, dev_ctms=dev_ctm_files, test_ctms=test_ctm_files)
        results[training_name + "_dev_all"] = dev_wer
        if test_wer is not None:
            results[training_name + "_test_all"] = test_wer

    assert not rtf_args or run_rtf is True
    if run_rtf is True:
        for key, tune_values in [("test", tune_values)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            if rtf_args.include_cpu is True:
                run_rtf_test(
                    search_name=training_name + f"/rtf_amd",
                    base_decoder_config=base_decoder_config,
                    lm_scales=[pick_optimal_params_job.out_optimal_parameters[0]],
                    prior_scales=[pick_optimal_params_job.out_optimal_parameters[1]],
                    dev_dataset_tuples=dev_dataset_tuples,
                    device="amd",
                    asr_model=asr_model,
                    rtf_args=rtf_args,
                    extra_forward_config=extra_forward_config or None,
                )
            if rtf_args.include_gpu is True:
                run_rtf_test(
                    search_name=training_name + f"/rtf_gpu",
                    base_decoder_config=base_decoder_config,
                    lm_scales=[pick_optimal_params_job.out_optimal_parameters[0]],
                    prior_scales=[pick_optimal_params_job.out_optimal_parameters[1]],
                    dev_dataset_tuples=dev_dataset_tuples,
                    device="gpu_24gb",
                    asr_model=asr_model,
                    rtf_args=rtf_args,
                    use_gpu=True,
                    extra_forward_config=extra_forward_config or None,
                )
    if get_best_params is True:
        pick_optimal_params_job = GetOptimalParametersAsVariableJob(
            parameters=tune_parameters, values=tune_values, mode="minimize"
        )
        pick_optimal_params_job.add_alias(training_name + f"/pick_best_get")
    return results, pick_optimal_params_job


def run_rtf_test(
    search_name: str,
    base_decoder_config: Union[DecoderConfig, RasrDecoderConfig],
    lm_scales: List[float],
    prior_scales: List[float],
    dev_dataset_tuples: Dict[str, Any],
    asr_model: ASRModel,
    device: str,
    import_memristor: bool = False,
    extra_forward_config: Optional[dict[str, Any]] = None,
    use_gpu: bool = False,
    debug: bool = False,
    rtf_args: Optional[Union[RTFArgs, RasrRTFArgs]] = None,
):
    assert len(lm_scales) == len(prior_scales) == 1, "Currently only support for one scale"
    if isinstance(rtf_args, RTFArgs):
        if rtf_args.forward_args is not None:
            extra_forward_config = {**(extra_forward_config or {}), **rtf_args.forward_args}
        if rtf_args.type == "greedy":
            report = {}
            from .pytorch_networks.ctc.decoder.greedy_bpe_ctc_rescale_measure_v2 import DecoderConfig

            decoder_module = rtf_args.decoder_module or "ctc.greedy_bpe_ctc_rescale_measure_v1"
            for turn_off_quant in [False, "leave_as_is"]:
                decoder_config = DecoderConfig(
                    returnn_vocab=base_decoder_config.returnn_vocab,
                    energy_device=device,
                    turn_off_quant=turn_off_quant,
                )
                s = "quant" if turn_off_quant == False else "full"
                name = search_name + "/" + s
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
                tk.register_output(search_name + f"/wer_{s}", list(wers.values())[0])
                report[s] = (
                    search_jobs[0].out_files["rtf"],
                    search_jobs[0].out_files["energy"],
                    list(wers.values())[0],
                )

            tk.register_report(
                f"reports/{search_name.split('/')[-1]}/{search_name.split('/')[-4]}_greedy",
                partial(build_greedy_rtf_report, report),
            )
        elif rtf_args.type == "nn_lm":
            raise NotImplementedError
            report = {}
            from .pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v4_rescale_measure import DecoderConfig
            from .pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v4_rescale_measure import DecoderExtraConfig
            from ... import PACKAGE

            decoder_module = rtf_args.decoder_module or "ctc.beam_search_bpe_ctc_v4_rescale_measure"
            for turn_off_quant in [False, "leave_as_is"]:
                decoder_config = DecoderConfig(
                    **asdict(base_decoder_config),
                    energy_device=device,
                    turn_off_quant=turn_off_quant,
                )
                decoder_config.lm_weight = lm_scales[0]
                decoder_config.prior_scale = prior_scales[0]
                decoder_unhashed_config_v3 = DecoderExtraConfig(
                    lm_package=PACKAGE,
                )
                s = "quant" if turn_off_quant == False else "full"
                name = search_name + "/" + s
                search_jobs: List[ReturnnForwardJobV2]
                search_jobs, wers = search(
                    name,
                    forward_config=extra_forward_config or {"num_workers_per_gpu": 0},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    unhashed_decoder_args={"extra_config": asdict(decoder_unhashed_config_v3)},
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
                tk.register_output(search_name + f"/wer_{s}", list(wers.values())[0])
                report[s] = (
                    search_jobs[0].out_files["rtf"],
                    search_jobs[0].out_files["energy"],
                    list(wers.values())[0],
                )
            tk.register_report(
                f"reports/{search_name.split('/')[-4]}/{search_name.split('/')[-3]}/{search_name.split('/')[-1]}_nn_lm",
                partial(build_nnlm_rtf_report, report),
            )
        else:
            decoder_module = rtf_args.decoder_module or "ctc.decoder.flashlight_ctc_v6_rescale_measure"
            from .pytorch_networks.ctc.decoder.flashlight_ctc_v7_rescale_measure import DecoderConfig
            from .pytorch_networks.ctc.decoder.flashlight_ctc_v7_rescale_measure import (
                DecoderConfig as QuantDecoderConfig,
            )

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
                                    additional_outputs=["rtf", "energy"] if not "v8" in decoder_module else ["rtf", "energy", "energy_sofware"],
                                    **default_returnn,
                                )
                                for job in search_jobs:
                                    job.rqmt["sbatch_args"] = f"-p rescale_{device} -A rescale_speed"
                                    job.rqmt["cpu"] = 2
                                assert len(search_jobs) == 1, "Only one search job is supported for now"
                                # tk.register_output(
                                #     search_name + f"/rtf_{beam_size}_{beam_size_token}_{beam_threshold}",
                                #     search_jobs[0].out_files["rtf"],
                                # )
                                # tk.register_output(
                                #     search_name + f"/energy_{beam_size}_{beam_size_token}_{beam_threshold}",
                                #     search_jobs[0].out_files["energy"],
                                # )
                                tk.register_output(
                                    search_name + f"/wer_{beam_size}_{beam_size_token}_{beam_threshold}",
                                    list(wers.values())[0],
                                )
                                report[f"{beam_size}_{beam_size_token}_{beam_threshold}"] = (
                                    search_jobs[0].out_files["rtf"],
                                    search_jobs[0].out_files["energy"],
                                    list(wers.values())[0],
                                )
            tk.register_report(
                f"reports/{search_name.split('/')[-1]}/{search_name.split('/')[-3]}", partial(build_rtf_report, report)
            )
            if not rtf_args.run_quant:
                return
            report = {}
            for lm_weight in lm_scales:
                for prior_scale in prior_scales:
                    beam_sizes = rtf_args.beam_sizes or [base_decoder_config.beam_size]
                    beam_size_tokens = rtf_args.beam_size_tokens or [base_decoder_config.beam_size_token]
                    beam_thresholds = rtf_args.beam_thresholds or [base_decoder_config.beam_threshold]
                    for beam_size in beam_sizes:
                        for beam_size_token in beam_size_tokens:
                            for beam_threshold in beam_thresholds:
                                decoder_config = QuantDecoderConfig(
                                    beam_size=beam_size,
                                    beam_size_token=beam_size_token,
                                    beam_threshold=beam_threshold,
                                    lm_weight=lm_weight,
                                    prior_scale=prior_scale,
                                    lexicon=base_decoder_config.lexicon,
                                    returnn_vocab=base_decoder_config.returnn_vocab,
                                    energy_device=device,
                                    arpa_lm=base_decoder_config.arpa_lm,
                                    turn_off_quant=False,
                                )
                                name = search_name + f"/{beam_size}_{beam_size_token}_{beam_threshold}_quantized"
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
                                # tk.register_output(
                                #     search_name + f"/rtf_{beam_size}_{beam_size_token}_{beam_threshold}",
                                #     search_jobs[0].out_files["rtf"],
                                #     )
                                # tk.register_output(
                                #     search_name + f"/energy_{beam_size}_{beam_size_token}_{beam_threshold}",
                                #     search_jobs[0].out_files["energy"],
                                #     )
                                tk.register_output(
                                    search_name + f"/wer_{beam_size}_{beam_size_token}_{beam_threshold}_quant",
                                    list(wers.values())[0],
                                )
                                report[f"{beam_size}_{beam_size_token}_{beam_threshold}"] = (
                                    search_jobs[0].out_files["rtf"],
                                    search_jobs[0].out_files["energy"],
                                    list(wers.values())[0],
                                )
            tk.register_report(
                f"reports/{search_name.split('/')[-1]}/{search_name.split('/')[-3]}_quantized",
                partial(build_rtf_report, report),
            )
    else:
        if rtf_args.type == "greedy":
            raise NotImplementedError
        elif rtf_args.type == "nn_lm":
            raise NotImplementedError
        else:
            decoder_module = rtf_args.decoder_module or "ctc.decoder.rasr_ctc_v1_rescale_measure_v1"
            from .pytorch_networks.ctc.decoder.rasr_ctc_v1_rescale_measure_v2 import DecoderConfig
            quant_ls = ["leave_as_is"]
            if rtf_args.run_quant:
                quant_ls.append(False)
            for turn_off_quant in quant_ls:
                s = "/quant" if turn_off_quant == False else "/full"
                report = {}
                for lm_weight in lm_scales:
                    for prior_scale in prior_scales:
                        max_beam_sizes = rtf_args.max_beam_size or [
                            base_decoder_config.rasr_config_file.lib_rasr.search_algorithm.max_beam_size
                        ]
                        score_thresholds = rtf_args.score_threshold or [
                            base_decoder_config.rasr_config_file.lib_rasr.search_algorithm.score_threshold
                        ]
                        for max_beam_size in max_beam_sizes:
                            for score_threshold in score_thresholds:
                                from .pytorch_networks.ctc.decoder.rasr_ctc_v1_rescale_measure_v2 import DecoderConfig
                                decoder_config = copy.deepcopy(base_decoder_config)
                                decoder_config.rasr_config_file.lib_rasr.search_algorithm.max_beam_size = max_beam_size
                                decoder_config.rasr_config_file.lib_rasr.search_algorithm.score_threshold = (
                                    score_threshold
                                )
                                decoder_config.rasr_config_file.lib_rasr.lm.scale = lm_weight
                                recog_rasr_config_path = WriteRasrConfigJob(
                                    decoder_config.rasr_config_file, decoder_config.rasr_post_config
                                ).out_config
                                decoder_config = DecoderConfig(
                                    energy_device=device,
                                    rasr_config_file=recog_rasr_config_path,
                                    rasr_post_config=None,
                                    blank_log_penalty=decoder_config.blank_log_penalty,
                                    prior_scale=prior_scale,
                                    prior_file=decoder_config.prior_file,
                                    turn_off_quant=turn_off_quant,
                                )
                                name = search_name + s + f"/{max_beam_size}_{score_threshold}"
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
                                    run_rasr=True,
                                    **default_returnn,
                                )
                                for job in search_jobs:
                                    job.rqmt["sbatch_args"] = f"-p rescale_{device} -A rescale_speed"
                                    job.rqmt["cpu"] = 2
                                assert len(search_jobs) == 1, "Only one search job is supported for now"
                                # tk.register_output(
                                #     search_name + f"/rtf_{beam_size}_{beam_size_token}_{beam_threshold}",
                                #     search_jobs[0].out_files["rtf"],
                                # )
                                # tk.register_output(
                                #     search_name + f"/energy_{beam_size}_{beam_size_token}_{beam_threshold}",
                                #     search_jobs[0].out_files["energy"],
                                # )
                                tk.register_output(
                                    search_name + f"/wer_{max_beam_size}_{score_threshold}" + s, list(wers.values())[0]
                                )
                                report[f"{max_beam_size}_{score_threshold}"] = (
                                    search_jobs[0].out_files["rtf"],
                                    search_jobs[0].out_files["energy"],
                                    list(wers.values())[0],
                                )
                tk.register_report(
                    f"reports/{search_name.split('/')[-1]}/{search_name.split('/')[-3]}{s}",
                    partial(build_rasr_rtf_report, report),
                )


from i6_core.util import instanciate_delayed


def build_report(report: Dict):
    report = copy.deepcopy(report)
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
            best_baselines[" ".join(exp.split("/")[4:])] = "{:.2f}".format(float(dic[best]))
        else:
            best_baselines[" ".join(exp.split("/")[4:])] = "None"
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_dc[" ".join(exp.split("/")[4:])] = "{:.2f}".format(float(dic[best]))
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
    report = copy.deepcopy(report)
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


def build_base_report(report: Dict, print_larger_params: bool = True):
    import numpy as np
    best_dc = {}
    report = copy.deepcopy(report)
    for exp, dic in report.items():
        instanciate_delayed(dic)
        # tmp = {x: dic[x] for x in dic.keys() if not "test" in x}
        if all(dic.values()):
            if len(dic) == 0:
                assert False, exp
            if "cycle" in exp:
                test_ls = []
                dev_ls = []
                general_ls = []
                for item in dic:
                    if "test_all" in item:
                        test_ls.append(dic[item])
                    elif "dev_al" in item:
                        dev_ls.append(dic[item])
                    else:
                        general_ls.append(dic[item])
                for ls in [general_ls, dev_ls, test_ls]:
                    if len(ls) == 0:
                        continue
                    mean = np.round(np.mean(ls), decimals=2)
                    mini = np.min(ls)
                    maxi = np.max(ls)
                    std = np.round(np.std(ls), decimals=4)
                    ln = len(ls)
                    if ls == test_ls:
                        st = "_test_all"
                    elif ls == dev_ls:
                        st = "_dev_all"
                    else:
                        st = ""
                    best = str(min(ls))
                    best_dc[" ".join(exp.split("/")[5:]) + st] = (
                        "{:.1f}".format(mean),
                        best + f"/{mean=} {std=} min: {'{:.1f}'.format(mini)} max: {'{:.1f}'.format(maxi)} runs: {ln}",
                    )
            else:
                best = min(dic, key=dic.get)
                best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(dic[best])), best)
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []

    for exp, value in best_dc.items():
        if any(y in exp for y in ["larger_search", "largerer_search", "full_dev", "full_test", "commonvoice", "librispeech", "voxpopuli", "yodas"]):
            continue

        ln = f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[8:10])}"
        wers = []
        if exp + "_full_dev" in best_dc:
            wers.append(best_dc[exp + "_full_dev"][0])
            wers.append(
                " ".join(
                    ("(CV:", best_dc[exp + "_dev_commonvoice"][0], "LBS:", best_dc[exp + "_dev_librispeech"][0], "VP:",
                        best_dc[exp + "_dev_voxpopuli"][0], "YD:", best_dc[exp + "_dev_yodas"][0], ")     ")))
        else:
            wers.append("None")
            # if not "cycle" in exp:
            #     assert False, (exp, best_dc)
        if exp + "_full_test" in best_dc:
            wers.append(best_dc[exp + "_full_test"][0])
            wers.append(
                " ".join(("(CV:", best_dc[exp + "_test_commonvoice"][0], "LBS:", best_dc[exp + "_test_librispeech"][0],
                    "VP:", best_dc[exp + "_test_voxpopuli"][0], "YD:", best_dc[exp + "_test_yodas"][0], ")     ")))
        else:
            wers.append("None")

        ln += f"  {', '.join(wers)}"
        line.append(ln)

    return "\n".join(line)


def build_greedy_rtf_report(report: Dict):
    report = copy.deepcopy(report)
    instanciate_delayed(report)
    line = []

    line.append("Name".ljust(7) + "WER".ljust(7) + "AM RTF".ljust(12) + "Energy".ljust(12))

    for res in report:
        if os.path.exists(report[res][0]):
            rtf = open(report[res][0]).read()
            am_rtf = rtf.split(",")[1].split(":")[1].split("\n")[0].strip()
        else:
            am_rtf = "None"
        if os.path.exists(report[res][1]):
            energy = float(open(report[res][1], "rt").read()) / 3600
        else:
            energy = "0"
        wer = report[res][2] or "0"

        line.append(
            f"{res}".ljust(7)
            + f"{wer}".ljust(7)
            + f"{am_rtf} ".ljust(12)
            + f"{float(energy):.2f}".ljust(12)
            + f"{float(wer):.1f}".ljust(7)
            + f"{float(wer):.2f}".ljust(7)
        )
    return "\n".join(line)


def build_nnlm_rtf_report(report: Dict):

    report = copy.deepcopy(report)
    instanciate_delayed(report)

    line = []
    line.append(
        "Name".ljust(7)
        + "WER".ljust(7)
        + "Search RTF".ljust(12)
        + "AM RTF".ljust(12)
        + "LM RTF".ljust(12)
        + "Energy".ljust(12)
        + "Total".ljust(7)
    )
    for res in report:
        if os.path.exists(report[res][0]):
            rtf = open(report[res][0]).read()
            search_rtf = rtf.split(",")[2].split(":")[1].split("\n")[0].strip()
            am_rtf = rtf.split(",")[1].split(":")[1].split("\n")[0].strip()
            total_rtf = rtf.split(",")[4].split(":")[1].split("\n")[0].strip()
            lm_rtf = rtf.split(",")[5].split(":")[1].split("\n")[0].strip()
        else:
            search_rtf = "None"
            am_rtf = "None"
            total_rtf = "None"
            lm_rtf = "None"
        if os.path.exists(report[res][1]):
            energy = float(open(report[res][1], "rt").read()) / 3600
        else:
            energy = "0"
        wer = report[res][2] or "0"

        line.append(
            f"{res}".ljust(7)
            + f"{wer}".ljust(7)
            + f"{search_rtf} ".ljust(12)
            + f"{am_rtf} ".ljust(12)
            + f"{lm_rtf} ".ljust(12)
            + f"{total_rtf} ".ljust(12)
            + f"{float(energy):.2f}".ljust(12)
            + f"{float(wer):.1f}".ljust(7)
            + f"{float(wer):.2f}".ljust(7)
        )
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
        + "AM RTF".ljust(12)
        + "Total".ljust(12)
        + "Energy".ljust(12)
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
                    + f"{am_rtf} ".ljust(12)
                    + f"{total_rtf} ".ljust(12)
                    + f"{float(energy):.2f}".ljust(12)
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
                        + f"{am_rtf} ".ljust(12)
                        + f"{total_rtf} ".ljust(12)
                        + f"{float(energy):.2f}".ljust(12)
                        + f"{float(wer):.1f}".ljust(7)
                        + f"{float(wer):.2f}".ljust(7)
                    )
                    min_score = wer

    line.insert(0, min_line)
    line.insert(1, " ")
    return "\n".join(line)


def build_rasr_rtf_report(report: Dict):
    max_beam_sizes = set()
    score_thresholds = set()

    report = copy.deepcopy(report)
    instanciate_delayed(report)
    for exp in report:
        beam, thresh = exp.split("_")
        max_beam_sizes.add(int(beam))
        score_thresholds.add(float(thresh))

    line = []
    min_line = ""
    min_score = 10000
    line.append(
        "Beam".ljust(7)
        + "Thresh".ljust(7)
        + "WER".ljust(7)
        + "Search RTF".ljust(12)
        + "AM RTF".ljust(12)
        + "Total".ljust(12)
        + "Energy".ljust(12)
    )
    for beam in sorted(max_beam_sizes):
        for thresh in sorted(score_thresholds):
            if os.path.exists(report[f"{beam}_{thresh}"][0]):
                rtf = open(report[f"{beam}_{thresh}"][0], "rt").read()
                search_rtf = rtf.split(",")[2].split(":")[1].split("\n")[0].strip()
                am_rtf = rtf.split(",")[1].split(":")[1].split("\n")[0].strip()
                total_rtf = rtf.split(",")[4].split(":")[1].split("\n")[0].strip()
            else:
                search_rtf = "None"
                am_rtf = "None"
                total_rtf = "None"

            if os.path.exists(report[f"{beam}_{thresh}"][1]):
                energy = float(open(report[f"{beam}_{thresh}"][1], "rt").read()) / 3600
            else:
                energy = "0"
            wer = report[f"{beam}_{thresh}"][2] or "0"

            line.append(
                f"{beam}".ljust(7)
                + f"{thresh}".ljust(7)
                + f"{wer}".ljust(7)
                + f"{search_rtf} ".ljust(12)
                + f"{am_rtf} ".ljust(12)
                + f"{total_rtf} ".ljust(12)
                + f"{float(energy):.2f}".ljust(12)
                + f"{float(wer):.1f}".ljust(7)
                + f"{float(wer):.2f}".ljust(7)
            )
            if isinstance(wer, float) and wer < min_score:
                min_line = (
                    f"{beam}".ljust(7)
                    + f"{thresh}".ljust(7)
                    + f"{wer}".ljust(7)
                    + f"{search_rtf} ".ljust(12)
                    + f"{am_rtf} ".ljust(12)
                    + f"{total_rtf} ".ljust(12)
                    + f"{float(energy):.2f}".ljust(12)
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
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.2f}".format(float(tmp[best])), best)
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


def build_qat_report(report: Dict, print_larger_params: bool = True):
    import numpy as np

    exps = ["batched", "combined", "fixed", "bal", "posadc", "learnpos", "keep_encs", "nolinpos", "cycle", "smaller", "greedy", "quant_out", "pertensor"]
    report = copy.deepcopy(report)
    best_dc = {}
    bits = [8, 7, 6, 5, 4, 3, 2, 1.5]

    import re as _re

    def _exp_tokens(exp):
        """Normalised identifying tokens for matching cycle keys to non-memristor keys.

        Handles both _W_A_seed_S and _wW_aA_seed_S bit-notation styles.
        Ignores decoding-only suffixes like _p05_l1 that do not affect the trained model.
        """
        part = exp.split("/")[-1]
        tokens = set()
        m = _re.search(r'seed_(\d+)', part)
        if m:
            tokens.add(f"seed_{m.group(1)}")
        m = _re.search(r'(\d+)eps', part)
        if m:
            tokens.add(f"{m.group(1)}eps")
        m = _re.search(r'(\d+)dim', part)
        if m:
            tokens.add(f"{m.group(1)}dim")
        # Normalise bit notation: w4_a8 and _4_8_ both become w4/a8 tokens
        m = _re.search(r'w(\d+)_a(\d+)', part)
        if m:
            tokens.add(f"w{m.group(1)}")
            tokens.add(f"a{m.group(2)}")
        else:
            m = _re.search(r'_(\d+)_(\d+)(?:_pertensor|_ideal|_seed)', part)
            if m:
                tokens.add(f"w{m.group(1)}")
                tokens.add(f"a{m.group(2)}")
        # Training-relevant modifiers
        if 'ideal_correct' in part:
            tokens.add('ideal_correct')
        elif 'ideal' in part:
            tokens.add('ideal')
        if 'pertensor' in part:
            tokens.add('pertensor')
        if 'batched' in part:
            tokens.add('batched')
        return frozenset(tokens)

    # Pre-pass: collect non-memristor WERs keyed by (6-part path prefix, token frozenset).
    # Cycle results are then compared against the same trained model run without memristor hardware.
    non_mem_wers = {}
    for exp, dic in report.items():
        if "cycle" in exp:
            continue
        tokens = _exp_tokens(exp)
        if not tokens:
            continue
        if "_full_dev" in exp:
            suffix = "dev"
        elif "_full_test" in exp:
            suffix = "test"
        elif not any(x in exp for x in ["_full_dev", "_full_test", "_dev_", "_test_"]):
            suffix = "general"
        else:
            continue
        instanciate_delayed(dic)
        vals = [v for v in dic.values() if v is not None]
        if not vals:
            continue
        try:
            key = ("/".join(exp.split("/")[:6]), tokens)
            if key not in non_mem_wers:
                non_mem_wers[key] = {}
            non_mem_wers[key][suffix] = float(min(vals))
        except (TypeError, ValueError):
            pass

    def _find_non_mem(prefix, cycle_tokens):
        """Return WER dict for the best-matching non-memristor experiment.

        Picks the non-memristor entry whose token set is a superset of cycle_tokens,
        preferring the most specific (smallest) superset to avoid false matches.
        """
        candidates = [
            (tokens, wers) for (p, tokens), wers in non_mem_wers.items()
            if p == prefix and cycle_tokens.issubset(tokens)
        ]
        if not candidates:
            return {}
        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0][1]

    for exp, dic in report.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            if "cycle" in exp:
                test = {}
                dev = {}
                general = {}
                for item in dic:
                    if "test_all" in item:
                        test[item] = dic[item]
                    elif "dev_all" in item:
                        dev[item] = dic[item]
                    else:
                        general[item] = dic[item]
                prefix = "/".join(exp.split("/")[:6])
                non_mem = _find_non_mem(prefix, _exp_tokens(exp))
                for dc in [general, dev, test]:
                    if len(dc) == 0:
                        continue
                    mean = np.round(np.mean(list(dc.values())), decimals=2)
                    mini = np.min(list(dc.values()))
                    maxi = np.max(list(dc.values()))
                    std = np.round(np.std(list(dc.values())), decimals=4)
                    ln = len(dc.values())
                    if dc == test:
                        st = "_test_all"
                        bline = non_mem.get("test")
                    elif dc == dev:
                        st = "_dev_all"
                        bline = non_mem.get("dev")
                    else:
                        st = ""
                        bline = non_mem.get("general")
                    best = min(dc, key=dc.get)
                    if bline is not None and bline > 0:
                        rel_dev = (float(mean) - bline) / bline * 100
                        rel_str = " ({:+.1f}% vs Base ({:.1f}%))".format(rel_dev, bline)
                    else:
                        rel_str = ""
                    best_dc[" ".join(exp.split("/")[5:]) + st] = (
                        "{:.1f} $\pm$ {:.2f} & [{:.1f}, {:.1f}]{}".format(mean, std, mini, maxi, rel_str),
                        best + f"/{mean=} {std=} min: {'{:.1f}'.format(mini)} max: {'{:.1f}'.format(maxi)} runs: {ln}",
                    )
            else:
                best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(dic[best])), best)
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    tmp = copy.deepcopy(best_dc)

    first = True
    for exp, value in best_dc.items():
        if "baseline" in exp:
            if any(y in exp for y in ["larger_search", "largerer_search", "full_dev", "full_test"]):
                continue
            if any(y in exp for y in ['commonvoice', 'librispeech', 'voxpopuli', 'yodas']):
                continue
            if first is True:
                line.append("Baseline")
                line.append("")
                first = False
            ln = f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[8:])}"
            wers = []
            if exp + "_full_dev" in best_dc:
                wers.append(best_dc[exp + "_full_dev"][0])
                if exp + "_dev_commonvoice" in best_dc:
                    wers.append(
                        " ".join(("(CV:", best_dc[exp + "_dev_commonvoice"][0], "LBS:",
                            best_dc[exp + "_dev_librispeech"][0], "VP:",
                            best_dc[exp + "_dev_voxpopuli"][0], "YD:", best_dc[exp + "_dev_yodas"][0],
                            ")     ")))
                    del tmp[exp + "_dev_commonvoice"]
                    del tmp[exp + "_dev_librispeech"]
                    del tmp[exp + "_dev_voxpopuli"]
                    del tmp[exp + "_dev_yodas"]
                del tmp[exp + "_full_dev"]
            else:
                wers.append("None")
                # if not "cycle" in exp:
                #     assert False, (exp, best_dc)
            if exp + "_full_test" in best_dc:
                wers.append(best_dc[exp + "_full_test"][0])
                if exp + "_test_commonvoice" in best_dc:
                    wers.append(
                        " ".join(("(CV:", best_dc[exp + "_test_commonvoice"][0], "LBS:",
                            best_dc[exp + "_test_librispeech"][0],
                            "VP:", best_dc[exp + "_test_voxpopuli"][0], "YD:", best_dc[exp + "_test_yodas"][0],
                            ")     ")))
                    del tmp[exp + "_test_commonvoice"]
                    del tmp[exp + "_test_librispeech"]
                    del tmp[exp + "_test_voxpopuli"]
                    del tmp[exp + "_test_yodas"]
                del tmp[exp + "_full_test"]
            else:
                wers.append("None")
            ln += f"  {', '.join(wers)}"
            line.append(ln)
            del tmp[exp]

    if first == False:
        line.append("")
    for bit in bits:
        best_dc = tmp
        tmp = copy.deepcopy(best_dc)
        first = False
        for exp, value in best_dc.items():
            if any(x in exp for x in exps):
                continue
            if any(y in exp for y in ["larger_search", "largerer_search", "full_dev", "full_test", "commonvoice", "librispeech", "voxpopuli", "yodas"]):
                continue
            if f"{bit}_8" not in exp and f"w{bit}_a8" not in exp:
                continue
            ln = (f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[8:])}")
            wers = []
            if exp + "_full_dev" in best_dc:
                wers.append(best_dc[exp + "_full_dev"][0])
                if exp + "_dev_commonvoice" in best_dc:
                    wers.append(
                        " ".join(("(CV:", best_dc[exp + "_dev_commonvoice"][0], "LBS:", best_dc[exp + "_dev_librispeech"][0], "VP:",
                         best_dc[exp + "_dev_voxpopuli"][0], "YD:", best_dc[exp + "_dev_yodas"][0], ")     ")))
                    del tmp[exp + "_dev_commonvoice"]
                    del tmp[exp + "_dev_librispeech"]
                    del tmp[exp + "_dev_voxpopuli"]
                    del tmp[exp + "_dev_yodas"]
                del tmp[exp + "_full_dev"]
            else:
                wers.append("None")
                # if not "cycle" in exp:
                #     assert False, (exp, best_dc)
            if exp + "_full_test" in best_dc:
                wers.append(best_dc[exp + "_full_test"][0])
                if exp + "_test_commonvoice" in best_dc:
                    wers.append(
                        " ".join(("(CV:", best_dc[exp + "_test_commonvoice"][0], "LBS:", best_dc[exp + "_test_librispeech"][0],
                         "VP:", best_dc[exp + "_test_voxpopuli"][0], "YD:", best_dc[exp + "_test_yodas"][0], ")     ")))
                    del tmp[exp + "_test_commonvoice"]
                    del tmp[exp + "_test_librispeech"]
                    del tmp[exp + "_test_voxpopuli"]
                    del tmp[exp + "_test_yodas"]
                del tmp[exp + "_full_test"]
            else:
                wers.append("None")
            ln += f"  {', '.join(wers)}"
            line.append(ln)
            if first == False:
                first = True
            del tmp[exp]
        if first == True:
            line.append("")
    tmp = best_dc
    for x in exps:
        first = True
        best_dc = copy.deepcopy(tmp)
        if x == "noise":
            w_bits = set()
            a_bits = set()
            starts = set()
            devs = set()
            dropouts = set()
            for exp in best_dc:
                if "noise" in exp:
                    pref = "_".join(exp.split("_")[:3])
                    w_bits.add(exp.split("_")[3])
                    a_bits.add(exp.split("_")[4])
                    starts.add(exp.split("_")[5][len("noise") :])
                    devs.add(exp.split("_")[6])
                    dropouts.add(exp.split("_")[7].split(" ")[0][len("drop") :])
            if all(len(x) == 0 for x in [w_bits, a_bits, starts, devs, dropouts]):
                continue
            line.append(x)
            line.append(
                "Weight B".ljust(10)
                + "Start".ljust(10)
                + "Dropout".ljust(10)
                + "Deviation".ljust(10)
                + "No Noise".ljust(10)
                + "Noise".ljust(10)
                + "Memristor".ljust(10)
            )
            for w_bit in sorted(w_bits, reverse=True):
                for a_bit in sorted(a_bits, reverse=True):
                    for start in sorted(starts):
                        for dev in sorted(devs):
                            for dropout in sorted(dropouts):
                                if (
                                    pref
                                    + "_"
                                    + w_bit
                                    + "_"
                                    + a_bit
                                    + "_noise"
                                    + start
                                    + "_"
                                    + dev
                                    + "_drop"
                                    + dropout
                                    + " without_noise"
                                    in best_dc
                                ):
                                    no_noise = best_dc[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_noise"
                                        + start
                                        + "_"
                                        + dev
                                        + "_drop"
                                        + dropout
                                        + " without_noise"
                                    ]
                                    del tmp[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_noise"
                                        + start
                                        + "_"
                                        + dev
                                        + "_drop"
                                        + dropout
                                        + " without_noise"
                                    ]
                                else:
                                    no_noise = "NaN"
                                if (
                                    pref
                                    + "_"
                                    + w_bit
                                    + "_"
                                    + a_bit
                                    + "_noise"
                                    + start
                                    + "_"
                                    + dev
                                    + "_drop"
                                    + dropout
                                    + " with_noise"
                                    in best_dc
                                ):
                                    noise = best_dc[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_noise"
                                        + start
                                        + "_"
                                        + dev
                                        + "_drop"
                                        + dropout
                                        + " with_noise"
                                    ]
                                    del tmp[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_noise"
                                        + start
                                        + "_"
                                        + dev
                                        + "_drop"
                                        + dropout
                                        + " with_noise"
                                    ]
                                else:
                                    noise = "NaN"
                                if (
                                    pref
                                    + "_"
                                    + w_bit
                                    + "_"
                                    + a_bit
                                    + "_noise"
                                    + start
                                    + "_"
                                    + dev
                                    + "_drop"
                                    + dropout
                                    + " cycle_combined"
                                    in best_dc
                                ):
                                    cycle = best_dc[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_noise"
                                        + start
                                        + "_"
                                        + dev
                                        + "_drop"
                                        + dropout
                                        + " cycle_combined"
                                    ]
                                    del tmp[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_noise"
                                        + start
                                        + "_"
                                        + dev
                                        + "_drop"
                                        + dropout
                                        + " cycle_combined"
                                    ]
                                else:
                                    cycle = "NaN"
                                if all(x == "NaN" for x in [no_noise, noise, cycle]):
                                    continue
                                line.append(
                                    f"{str(w_bit).ljust(10)}{str(start).ljust(10)}{str(dropout).ljust(10)}{str(dev).ljust(10)}{no_noise[0].ljust(10)}{noise[0].ljust(10)}{cycle[0].ljust(10)}{' '.join(cycle[1].split(' ')[1:]).ljust(25)} {' ' if len(cycle[1]) <= 1 else cycle[1].split('/')[-2]}"
                                )
            line.append("")
        elif x == "correction":
            w_bits = set()
            a_bits = set()
            cycles = set()
            devs = set()
            tests = set()
            for exp in best_dc:
                if "correction_" in exp and not "correction_baseline" in exp:
                    pref = "_".join(exp.split("_")[:3])
                    w_bits.add(exp.split("_")[3])
                    a_bits.add(exp.split("_")[4])
                    cycles.add(exp.split("_")[6])
                    devs.add(exp.split("_")[8][: -len(" cycle")])
                    tests.add(exp.split("_")[7])
            if all(len(x) == 0 for x in [w_bits, a_bits, cycles, devs, tests]):
                continue
            line.append(x)
            line.append(
                "Weight B".ljust(10)
                + "Cycles".ljust(10)
                + "Test Val".ljust(10)
                + "Deviation".ljust(10)
                + "Correction".ljust(10)
            )
            for w_bit in sorted(w_bits, reverse=True):
                for a_bit in sorted(a_bits, reverse=True):
                    line.append(
                        f"{w_bit.ljust(10)}Baseline: {best_dc[pref + '_' + w_bit + '_' + a_bit + '_correction_baseline'][0]}"
                    )
                    line.append(
                        f"{w_bit.ljust(10)}No Correction:      {best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][0]}                 {' '.join(best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][1].split(' ')[1:])} {best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][1].split('/')[-2] if len(best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][1].split('/')) > 1 else ''}"
                    )
                    del tmp[pref + "_" + w_bit + "_" + a_bit + "_correction_baseline"]
                    del tmp[pref + "_" + w_bit + "_" + a_bit + "_no_correction cycle_combined"]
                    for test_v in sorted(tests):
                        for dev in sorted(devs):
                            for cycle in sorted(cycles):
                                if (
                                    pref
                                    + "_"
                                    + w_bit
                                    + "_"
                                    + a_bit
                                    + "_correction_"
                                    + cycle
                                    + "_"
                                    + test_v
                                    + "_"
                                    + dev
                                    + " cycle_combined"
                                    in best_dc
                                ):
                                    mem = best_dc[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_correction_"
                                        + cycle
                                        + "_"
                                        + test_v
                                        + "_"
                                        + dev
                                        + " cycle_combined"
                                    ]
                                    del tmp[
                                        pref
                                        + "_"
                                        + w_bit
                                        + "_"
                                        + a_bit
                                        + "_correction_"
                                        + cycle
                                        + "_"
                                        + test_v
                                        + "_"
                                        + dev
                                        + " cycle_combined"
                                    ]
                                else:
                                    continue
                                line.append(
                                    f"{str(w_bit).ljust(10)}{str(cycle).ljust(10)}{str(test_v).ljust(10)}{str(dev).ljust(10)}{mem[0].ljust(10)}{' '.join(mem[1].split(' ')[1:])} {' ' if len(mem[1]) <= 1 else mem[1].split('/')[-2].ljust(25)}"
                                )
            line.append("")
        elif x == "baseline":
            for exp, value in best_dc.items():
                if x in exp:
                    if any(y in exp for y in ["larger_search", "largerer_search", "full_dev", "full_test"]):
                        continue
                    if any(y in exp for y in ['commonvoice', 'librispeech', 'voxpopuli', 'yodas']):
                        continue
                    if first is True:
                        line.append(x)
                        line.append("")
                        first = False
                    ln = f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[8:])}"
                    wers = []
                    if exp + "_full_dev" in best_dc:
                        wers.append(best_dc[exp + "_full_dev"][0])
                        if exp + "_dev_commonvoice" in best_dc:
                            wers.append(
                                " ".join(("(CV:", best_dc[exp + "_dev_commonvoice"][0], "LBS:",
                                    best_dc[exp + "_dev_librispeech"][0], "VP:",
                                    best_dc[exp + "_dev_voxpopuli"][0], "YD:", best_dc[exp + "_dev_yodas"][0],
                                    ")     ")))
                            del tmp[exp + "_dev_commonvoice"]
                            del tmp[exp + "_dev_librispeech"]
                            del tmp[exp + "_dev_voxpopuli"]
                            del tmp[exp + "_dev_yodas"]
                        del tmp[exp + "_full_dev"]
                    else:
                        wers.append("None")
                        # if not "cycle" in exp:
                        #     assert False, (exp, best_dc)
                    if exp + "_full_test" in best_dc:
                        wers.append(best_dc[exp + "_full_test"][0])
                        if exp + "_test_commonvoice" in best_dc:
                            wers.append(
                                " ".join(("(CV:", best_dc[exp + "_test_commonvoice"][0], "LBS:",
                                    best_dc[exp + "_test_librispeech"][0],
                                    "VP:", best_dc[exp + "_test_voxpopuli"][0], "YD:", best_dc[exp + "_test_yodas"][0],
                                    ")     ")))
                            del tmp[exp + "_test_commonvoice"]
                            del tmp[exp + "_test_librispeech"]
                            del tmp[exp + "_test_voxpopuli"]
                            del tmp[exp + "_test_yodas"]
                        del tmp[exp + "_full_test"]
                    else:
                        wers.append("None")
                    if not x in ["correction", "cycle"]:
                        ln += f"  {', '.join(wers)}"
                    line.append(ln)
                    del tmp[exp]
        else:
            for exp, value in best_dc.items():
                if x in exp:
                    if x == "baseline" and "correction_baseline" in exp:
                        continue
                    if any(y in exp for y in ["larger_search", "largerer_search"]):
                        continue
                    if first is True:
                        line.append(x)
                        line.append("")
                        first = False
                    ln = f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[8:])}"
                    line.append(ln)
                    del tmp[exp]
        if first is False:
            line.append("")
    assert len(tmp) == 0, tmp
    return "\n".join(line)

def build_qat_report_v2(report: Dict) -> str:
    """Tabular version of build_qat_report_v2. QAT and Memristor sections are rendered as aligned tables."""
    import numpy as np
    import re

    report = copy.deepcopy(report)

    DATASETS_FULL = ["commonvoice", "librispeech", "voxpopuli", "yodas"]
    DATASETS_SHORT = ["common", "librispeech", "voxpopuli", "yodas"]
    DS_LABEL = {"commonvoice": "CV", "common": "CV", "librispeech": "LBS", "voxpopuli": "VP", "yodas": "YD"}
    TYPE_KEYWORDS = [
        "posadc", "keep_encs", "nolinpos", "learnpos", "quant_out", "quantout",
        "pertensor", "batched", "combined", "fixed", "bal", "smaller", "greedy",
        "ideal_correct", "ideal", "adc",
    ]

    # ── token extraction & no-hw pre-pass (identical to v2) ──────────────────

    def exp_tokens(exp: str) -> frozenset:
        part = exp.split("/")[-1]
        tokens = set()
        for pattern, fmt in [(r"seed_(\d+)", "seed_{}"), (r"(\d+)eps", "{}eps"), (r"(\d+)dim", "{}dim")]:
            m = re.search(pattern, part)
            if m:
                tokens.add(fmt.format(m.group(1)))
        m = re.search(r"w([\dx_]+)_a(\d+)", part)
        if m:
            tokens.update({f"w{m.group(1)}", f"a{m.group(2)}"})
        else:
            m = re.search(r"_(\d+)_(\d+)(?:_pertensor|_ideal|_seed|_adc)", part)
            if m:
                tokens.update({f"w{m.group(1)}", f"a{m.group(2)}"})
        for mod in ("ideal_correct", "pertensor", "batched", "nolinpos", "learnpos", "keep_encs", "quantout", "quant_out", "greedy"):
            if mod in part:
                tokens.add(mod)
        if "ideal" in part and "ideal_correct" not in part:
            tokens.add("ideal")
        return frozenset(tokens)

    non_mem_wers: Dict = {}
    for exp, dic in report.items():
        if "cycle" in exp:
            continue
        if exp.endswith("_best") or exp.endswith("_best4"):
            continue
        tokens = exp_tokens(exp)
        if not tokens:
            continue
        if "_full_dev" in exp:
            suffix = "dev"
        elif "_full_test" in exp:
            suffix = "test"
        elif not any(s in exp for s in ("_full_dev", "_full_test", "_dev_", "_test_")):
            suffix = "general"
        else:
            continue
        instanciate_delayed(dic)
        vals = [v for k, v in dic.items() if v is not None and "/best" not in k]
        if not vals:
            continue
        try:
            key = ("/".join(exp.split("/")[:6]), tokens)
            non_mem_wers.setdefault(key, {})[suffix] = float(min(vals))
        except (TypeError, ValueError):
            pass

    HW_ONLY_TOKENS = frozenset({"ideal", "ideal_correct"})

    def find_non_mem(prefix: str, tokens: frozenset) -> Dict:
        lookup = tokens - HW_ONLY_TOKENS
        candidates = [(t, w) for (p, t), w in non_mem_wers.items()
                      if p == prefix and lookup.issubset(t)]
        return min(candidates, key=lambda x: len(x[0]))[1] if candidates else {}

    # ── build summary dict (identical to v2) ─────────────────────────────────

    def shorten(exp: str) -> str:
        return " ".join(exp.split("/")[5:])

    summary: Dict[str, tuple] = {}
    cycle_raw: Dict[str, Dict] = {}

    for exp, dic in report.items():
        short = shorten(exp)
        instanciate_delayed(dic)
        if "cycle" in exp:
            if not all(dic.values()):
                summary[short] = ("rng", "")
                continue
            groups = {"": {}, "_dev_all": {}, "_test_all": {}}
            for k, v in dic.items():
                if "test_all" in k:
                    groups["_test_all"][k] = v
                elif "dev_all" in k:
                    groups["_dev_all"][k] = v
                else:
                    groups[""][k] = v
            prefix = "/".join(exp.split("/")[:6])
            non_mem = find_non_mem(prefix, exp_tokens(exp))
            bline_by_st = {"": non_mem.get("general"), "_dev_all": non_mem.get("dev"), "_test_all": non_mem.get("test")}
            for st, dc in groups.items():
                if not dc:
                    continue
                vals = list(dc.values())
                mean = float(np.round(np.mean(vals), decimals=2))
                std = float(np.round(np.std(vals), decimals=4))
                mini, maxi = float(np.min(vals)), float(np.max(vals))
                best = min(dc, key=dc.get)
                bline = bline_by_st[st]
                rel = (mean - bline) / bline * 100 if (bline and bline > 0) else None
                rel_str = "" if rel is None else " ({:+.1f}% vs no-hw [{:.1f}])".format(rel, bline)
                wer_str = "{:.1f} \u00b1 {:.2f} [{:.1f},{:.1f}] n={}{}".format(mean, std, mini, maxi, len(vals), rel_str)
                detail = "best={} mean={:.2f} std={:.4f} min={:.1f} max={:.1f} n={}".format(
                    best, mean, std, mini, maxi, len(vals))
                summary[short + st] = (wer_str, detail)
                cycle_raw[short + st] = {"mean": mean, "std": std, "min": mini, "max": maxi, "n": len(vals), "rel": rel, "bline": bline}
        else:
            def _ckpt_min(d, tag):
                if tag == "best4":
                    group = {k: v for k, v in d.items() if "/best4" in k}
                elif tag == "best":
                    group = {k: v for k, v in d.items() if "/best" in k and "/best4" not in k}
                else:
                    group = {k: v for k, v in d.items() if "/best" not in k}
                if not group:
                    return "—"
                if not all(group.values()):
                    return "rng"
                return "{:.1f}".format(float(min(group.values())))
            has_best = any("/best" in k for k in dic)
            if has_best:
                parts = [_ckpt_min(dic, t) for t in ("last", "best", "best4")]
                summary[short] = ("/".join(parts), "")
            else:
                last_val = _ckpt_min(dic, "last")
                summary[short] = (last_val, "")

    # ── classification helpers (identical to v2) ──────────────────────────────

    CYCLE_STAT_SUFFIXES = {"_dev_all", "_test_all"}
    SPLIT_SUFFIXES = (
        {f"_full_{s}" for s in ("dev", "test")}
        | {f"_full_{s}_best" for s in ("dev", "test")}
        | {f"_{s}_{ds}" for s in ("dev", "test") for ds in DATASETS_FULL + DATASETS_SHORT}
        | {f"_{s}_{ds}_best" for s in ("dev", "test") for ds in DATASETS_FULL + DATASETS_SHORT}
    )
    DATASET_CYCLE_SUFFIXES = {f"_{ds}" for ds in ("yodas", "common", "commonvoice", "librispeech", "voxpopuli")}

    def is_sub_entry(key: str) -> bool:
        return any(key.endswith(s) for s in CYCLE_STAT_SUFFIXES | SPLIT_SUFFIXES | DATASET_CYCLE_SUFFIXES)

    def exp_label(exp_short: str) -> str:
        return " ".join(exp_short.split(".")[2:])

    def get_type(exp_short: str) -> str:
        for kw in TYPE_KEYWORDS:
            if kw in exp_short:
                return kw
        return "standard"

    def group_by_type(exps_dict: Dict) -> Dict[str, Dict]:
        groups: Dict[str, Dict] = {}
        for k, v in exps_dict.items():
            groups.setdefault(get_type(k), {})[k] = v
        return groups

    top = {k: v for k, v in summary.items() if not is_sub_entry(k)}
    baseline_exps = {k: v for k, v in top.items() if "baseline" in k}
    cycle_exps = {k: v for k, v in top.items() if "cycle" in k}
    qat_exps = {k: v for k, v in top.items() if k not in baseline_exps and k not in cycle_exps}

    # ── dump helpers (identical to v2) ────────────────────────────────────────

    def dataset_breakdown_for(exp_short: str, split: str) -> str:
        parts = []
        for ds_full, ds_short in zip(DATASETS_FULL, DATASETS_SHORT):
            for ds in (ds_full, ds_short):
                key = exp_short + f"_{split}_{ds}"
                if key in summary and summary[key][0] not in ("rng", "—"):
                    parts.append(f"{DS_LABEL[ds]}: {summary[key][0]}")
                    break
        return "  ".join(parts)

    def cycle_dataset_breakdown(exp_short: str) -> str:
        parts = []
        for ds in ("yodas", "common", "commonvoice", "librispeech", "voxpopuli"):
            key = exp_short + f"_{ds}"
            if key in summary and summary[key][0] not in ("rng", "—"):
                parts.append(f"{DS_LABEL.get(ds, ds.upper())}: {summary[key][0]}")
        return "  ".join(parts)

    def dump_lines_for(exp_short: str) -> List[str]:
        out = []
        label = exp_label(exp_short)
        for split in ("dev", "test"):
            bd = dataset_breakdown_for(exp_short, split)
            if bd:
                out.append(f"  {label} {split}: {bd}")
        cycle_bd = cycle_dataset_breakdown(exp_short)
        if cycle_bd:
            out.append(f"  {label} dev: {cycle_bd}")
        return out

    # ── table rendering ───────────────────────────────────────────────────────

    def make_table(headers: List[str], rows: List[List[str]]) -> List[str]:
        if not rows:
            return []
        all_rows = [headers] + rows
        widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
        sep = "-+-".join("-" * w for w in widths)
        def fmt(row):
            return " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
        return [fmt(headers), sep] + [fmt(r) for r in rows]

    def cell(exp_short: str, st: str) -> str:
        return summary.get(exp_short + st, ("—", ""))[0]

    def qat_table(exps_dict: Dict) -> List[str]:
        seed_groups: Dict[str, List[str]] = {}
        for e in exps_dict:
            base = re.sub(r"_seed_\d+", "", e)
            seed_groups.setdefault(base, []).append(e)

        MAX_SEEDS = 3
        seed_labels = [f"s{i}" for i in range(MAX_SEEDS)]
        splits = [("Short-dev", "", False), ("Full dev", "_full_dev", True), ("Full test", "_full_test", True)]
        n_splits = len(splits)

        def cell_val(e: str, st: str, has_best: bool) -> str:
            last = summary.get(e + st, ("—", ""))[0]
            if has_best:
                best = summary.get(e + st + "_best", ("—", ""))[0]
                if best != "—":
                    return f"{last}/{best}"
            return last

        data_rows = []
        for base, exps in seed_groups.items():
            seed_map = {}
            for e in exps:
                m = re.search(r"_seed_(\d+)", e)
                idx = int(m.group(1)) if m else 0
                seed_map[idx] = e
            row = [exp_label(base)]
            for _, st, hb in splits:
                for i in range(MAX_SEEDS):
                    e = seed_map.get(i)
                    row.append(cell_val(e, st, hb) if e else "—")
            data_rows.append(row)

        header2 = ["Experiment"] + seed_labels * n_splits
        all_rows = [header2] + data_rows
        widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header2))]

        header1_parts = [" " * widths[0]]
        for i, (split_name, _, _hb) in enumerate(splits):
            span = sum(widths[1 + i * MAX_SEEDS + j] for j in range(MAX_SEEDS)) + 3 * (MAX_SEEDS - 1)
            header1_parts.append(split_name.center(span))
        header1 = " | ".join(header1_parts)

        sep = "-+-".join("-" * w for w in widths)

        def fmt(row: List[str]) -> str:
            return " | ".join(str(c).ljust(w) for c, w in zip(row, widths))

        return [header1, fmt(header2), sep] + [fmt(r) for r in data_rows]

    def cycle_table(exps_dict: Dict) -> List[str]:
        splits = [("Short-dev", ""), ("Dev", "_dev_all"), ("Test", "_test_all")]
        has_rel = any(
            cycle_raw.get(e + st_sfx, {}).get("rel") is not None
            for e in exps_dict for _, st_sfx in splits
        )
        sub_cols = ["mean", "±std"] + (["rel%"] if has_rel else []) + ["min", "max", "n"]
        n_sub = len(sub_cols)

        def raw_cells(e: str, st_sfx: str) -> List[str]:
            r = cycle_raw.get(e + st_sfx)
            if not r:
                return ["—"] * n_sub
            cells = [
                "{:.1f}".format(r["mean"]),
                "{:.2f}".format(r["std"]),
            ]
            if has_rel:
                if r.get("rel") is not None:
                    cells.append("{:+.1f} ({:.1f})".format(r["rel"], r["bline"]))
                else:
                    cells.append("—")
            cells += [
                "{:.1f}".format(r["min"]),
                "{:.1f}".format(r["max"]),
                str(r["n"]),
            ]
            return cells

        data_rows = [[exp_label(e)] + [c for _, st_sfx in splits for c in raw_cells(e, st_sfx)]
                     for e in exps_dict]
        header2 = ["Experiment"] + sub_cols * len(splits)
        all_rows = [header2] + data_rows
        widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header2))]

        # first header row: blank exp column, then split name centred over its sub-columns
        header1_parts = [" " * widths[0]]
        for i, (split_name, _) in enumerate(splits):
            span = sum(widths[1 + i * n_sub + j] for j in range(n_sub)) + 3 * (n_sub - 1)
            header1_parts.append(split_name.center(span))
        header1 = " | ".join(header1_parts)

        sep = "-+-".join("-" * w for w in widths)

        def fmt(row: List[str]) -> str:
            return " | ".join(str(c).ljust(w) for c, w in zip(row, widths))

        return [header1, fmt(header2), sep] + [fmt(r) for r in data_rows]

    # ── render ────────────────────────────────────────────────────────────────

    lines = []
    dump: List[str] = []

    def render_table_section(title: str, exps_dict: Dict, table_fn) -> None:
        if not exps_dict:
            return
        lines.append(title)
        lines.append("=" * len(title))
        if table_fn is qat_table:
            lines.append("(WER cells: last epoch / best / best4)")
        groups = group_by_type(exps_dict)
        for gtype, entries in groups.items():
            if len(groups) > 1:
                lines.append("")
                lines.append(f"  -- {gtype} --")
                lines.append("")
            lines.extend(table_fn(entries))
            for exp_short in entries:
                dump.extend(dump_lines_for(exp_short))
        lines.append("")

    render_table_section("Baseline", baseline_exps, qat_table)
    render_table_section("QAT (no hardware)", qat_exps, qat_table)
    render_table_section("Memristor (hardware cycles)", cycle_exps, cycle_table)

    if dump:
        lines.append("Per-dataset results")
        lines.append("=" * len("Per-dataset results"))
        lines.extend(dump)

    return "\n".join(lines)
