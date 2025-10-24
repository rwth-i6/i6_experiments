import copy
from dataclasses import asdict
from typing import Optional, Dict, Any, List, Union

from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from ..data.dataset_commons import TrainingDatasets
from ..model_creation.pipeline import prepare_asr_model, ASRModel, search
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...recognition.beam_search import DecoderConfig

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
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
    loss_name: str = "dev_loss_ce",
    use_gpu: bool = False,
    extra_forward_config: Optional[dict[str, Any]] = None,

    # TO RUN FLAGS
    run_best_4: bool = True,
    run_best: bool = True,
    run_test: bool = False,

    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    prior_args: Optional[Dict[str, Any]] = None,
) -> Dict[Any, Any]:
    """
    What does this method do???

    """
    # DEFAULT PARAMETERS # TODO: MJ: defaults can be in parameters
    if specific_epoch is None:
        specific_epoch = train_job.returnn_config.post_config["num_epochs"]
    if isinstance(specific_epoch, int):
        specific_epoch = [specific_epoch]
    if lm_scales is None:
        lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    if prior_scales is None:
        prior_scales = [0.7, 0.9]
    debug = train_args.get("debug", False)

    result_dict = {} if result_dict is None else result_dict

    # TODO: Extract method to
        # create asr_model
        # check prior_args
        # tune_and_evaluate_helper
    for epoch in specific_epoch:
        asr_model = prepare_asr_model(
            training_name + f"/{epoch}",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=False,  # True,
            datasets=train_data,
            get_specific_checkpoint=epoch,
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
            debug=debug,
            extra_forward_config=extra_forward_config,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.dataset.target_options,
        )
        result_dict.update(res)

    if run_best_4:
        asr_model_best4 = prepare_asr_model(
            training_name + "/best4",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=False,  # True,
            datasets=train_data,
            get_best_averaged_checkpoint=(4, loss_name),
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
            debug=debug,
            extra_forward_config=extra_forward_config,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.dataset.target_options,
        )
        result_dict.update(res)

    if run_best:
        asr_model_best = prepare_asr_model(
            training_name + "/best",
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=False,  # True,
            datasets=train_data,
            get_best_averaged_checkpoint=(1, loss_name),
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
            debug=debug,
            extra_forward_config=extra_forward_config,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.dataset.target_options,
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
            tune_values_clean.append((wers[search_name + "/dev-clean"]))
            tune_values_other.append((wers[search_name + "/dev-other"]))
            results.update(wers)

    pick_optimal_params_job = None
    if run_test is True and test_dataset_tuples is not None:
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
                forward_config=extra_forward_config or {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
                vocab_opts=vocab_opts,
                debug=debug,
                **default_returnn,
            )
        results.update(wers)

    return results, pick_optimal_params_job