import copy
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple, List, Union

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from .asr_model_info import ASRModel
from .forward_job_builder import search, compute_prior
from ..data.dataset_commons import TrainingDatasets
from ..model_creation.returnn_config_helpers import get_prior_config
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...recognition.beam_search import DecoderConfig
from i6_core.returnn.training import ReturnnTrainingJob, AverageTorchCheckpointsJob, GetBestPtCheckpointJob, \
    PtCheckpoint

from sisyphus import tk

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def create_evaluation_jobs(
        training_name: str,
        train_job: ReturnnTrainingJob,
        train_args: Dict[str, Any],
        train_data: TrainingDatasets,
        decoder_config: DecoderConfig,

        dev_dataset_tuples: Dict[str, Any],
        test_dataset_tuples: Optional[Dict[str, Any]] = None,

        lm_scales: Optional[List[float]] = None,
        prior_scales: Optional[List[float]] = None,

        decoder_module: str = "should_not_be_default",
        loss_name: str = "dev_loss_ce",
        use_gpu: bool = False,
        extra_forward_config: Optional[dict[str, Any]] = None,
        prior_args: Optional[Dict[str, Any]] = None,

        # TO RUN FLAGS
        specific_epoch: Optional[Union[int, List]] = None,
        run_best_4: bool = True,
        run_best: bool = True,
        run_test: bool = False,

        result_dict: Optional[Dict[str, Any]] = None,
) -> Dict[Any, Any]:
    """
    Run evaluation jobs for different trained models

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

    # Return structure
    result_dict = {} if result_dict is None else result_dict

    # TODO: Extract method to
    # create asr_model
    # check prior_args
    # tune_and_evaluate_helper
    for epoch in specific_epoch:
        specific_training_name = training_name + f"/{epoch}"

        asr_model = prepare_asr_model(
            specific_training_name,
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=False,
            datasets=train_data,
            get_specific_checkpoint=epoch,
        )

        if prior_args is not None:
            asr_model.net_args = train_args["net_args"]
            asr_model.network_module = train_args["network_module"]

        res, _ = tune_and_evaluate_helper(
            specific_training_name,
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
        specific_training_name = training_name + "/best4"
        asr_model_best4 = prepare_asr_model(
            specific_training_name,
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=False,
            datasets=train_data,
            get_best_averaged_checkpoint=(4, loss_name),
        )
        if prior_args is not None:
            asr_model_best4.net_args = train_args["net_args"]
            asr_model_best4.network_module = train_args["network_module"]
        res, _ = tune_and_evaluate_helper(
            specific_training_name,
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
        specific_training_name = training_name + "/best"
        asr_model_best = prepare_asr_model(
            specific_training_name,
            train_job,
            train_args if prior_args is None else prior_args,
            with_prior=False,
            datasets=train_data,
            get_best_averaged_checkpoint=(1, loss_name),
        )
        if prior_args is not None:
            asr_model_best.net_args = train_args["net_args"]
            asr_model_best.network_module = train_args["network_module"]
        res, _ = tune_and_evaluate_helper(
            specific_training_name,
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


def prepare_asr_model(
        training_name: str,
        train_job: ReturnnTrainingJob,
        train_args: Dict[str, Any],
        with_prior: bool,

        datasets: Optional[TrainingDatasets] = None,
        get_specific_checkpoint: Optional[int] = None,
        get_best_averaged_checkpoint: Optional[Tuple[int, str]] = None,
        get_last_averaged_checkpoint: Optional[int] = None,
        prior_config: Optional[Dict[str, Any]] = None,
) -> ASRModel:
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

    checkpoint, training_name = get_checkpoint(get_best_averaged_checkpoint,
                                               get_last_averaged_checkpoint,
                                               get_specific_checkpoint,
                                               train_job,
                                               training_name)

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

    return ASRModel(
        checkpoint=checkpoint,
        network_module=train_args["network_module"],
        net_args=train_args["net_args"],
        prior_file=prior_file,
        prefix_name=training_name,
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





def get_checkpoint(
        get_best_averaged_checkpoint: tuple[int, str] | None,
        get_last_averaged_checkpoint: int | None,
        get_specific_checkpoint: int | None,
        train_job,
        training_name) -> tuple[Any, PtCheckpoint]:
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

    return checkpoint, training_name
