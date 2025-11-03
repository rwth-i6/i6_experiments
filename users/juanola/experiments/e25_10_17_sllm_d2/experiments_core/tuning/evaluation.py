import copy
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple, List, Union, Iterable

from sisyphus import tk, job_path

from i6_core.returnn.training import ReturnnTrainingJob, AverageTorchCheckpointsJob, GetBestPtCheckpointJob, \
    PtCheckpoint
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from .asr_model import ASRModel
from .forward_job_builder import search, compute_prior
from ..data.dataset_commons import TrainingDatasets
from ..model_creation.returnn_config_helpers import get_prior_config
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...recognition.decoder_config import DecoderConfig

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def create_tune_and_evaluate_jobs(
        training_name: str,
        train_job: ReturnnTrainingJob,

        network_module,
        net_args,
        debug,

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
        specific_epoch: Optional[Union[int, Iterable[int]]] = None,
        run_best_4: bool = True,
        run_best: bool = True,

        run_test: bool = False,

        result_dict: Optional[Dict[str, Any]] = None,
) -> Dict[Any, Any]:
    """
    Run evaluation jobs for different trained models

    """
    # DEFAULT PARAMETERS
    if specific_epoch is None:
        specific_epoch = train_job.returnn_config.post_config["num_epochs"]
    if isinstance(specific_epoch, int):
        specific_epoch = [specific_epoch]
    if lm_scales is None:
        lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    if prior_scales is None:
        prior_scales = [0.7, 0.9]

    # Dict of all train_evals to perform (could be extended)
    checkpoint_per_evaluation = OrderedDict()
    for epoch in specific_epoch:
        evaluation_name = f"{training_name}/{epoch}"
        checkpoint_per_evaluation[evaluation_name] = get_specific_checkpoint(evaluation_name, train_job, epoch)
    if run_best_4:
        evaluation_name = f"{training_name}/best4"
        checkpoint_per_evaluation[evaluation_name] = get_best_averaged_checkpoint(evaluation_name, train_job, 4,
                                                                                  loss_name)
    if run_best:
        evaluation_name = f"{training_name}/best"
        checkpoint_per_evaluation[evaluation_name] = get_best_averaged_checkpoint(evaluation_name, train_job, 1,
                                                                                  loss_name)

    # Tune & Eval different models
    result_dict = {} if result_dict is None else result_dict
    for evaluation_name, (checkpoint, checkpoint_name) in checkpoint_per_evaluation.items():
        asr_model = prepare_asr_model(
            checkpoint_name,
            checkpoint,
            network_module,
            net_args,
            prior_args=prior_args,
            datasets=train_data,
        )

        res, _ = tune_and_evaluate_model(
            evaluation_name,
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
    return result_dict


def prepare_asr_model(
        checkpoint_name: str,
        checkpoint: PtCheckpoint,
        network_module,
        net_args,
        prior_args: Dict[str, Any] = None,
        prior_config: Optional[Dict[str, Any]] = None,
        datasets: Optional[TrainingDatasets] = None,
) -> ASRModel:
    """
    :param checkpoint_name:
    :param checkpoint:
    :param network_module:
    :param net_args:
    :param prior_args:
    :param datasets: Needed if with_prior == True
    :param prior_config: if with_prior is true, can be used to add Returnn config parameters for the prior compute job
    :return:
    """
    assert prior_args is None or datasets is not None
    assert prior_args is None or prior_config is not None

    prior_file = None
    if prior_args is not None:
        returnn_config = get_prior_config(
            training_datasets=datasets,
            network_module=prior_args["network_module"],
            config=prior_config if prior_config is not None else {},
            net_args=prior_args["net_args"],
            unhashed_net_args=prior_args.get("unhashed_net_args", None),
            debug=prior_args.get("debug", False),
        )
        prior_file = compute_prior(
            checkpoint_name,
            returnn_config,
            checkpoint=checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
        )
        tk.register_output(f"{checkpoint_name}/prior.txt", prior_file)
    else:
        if prior_config is not None:
            raise ValueError("prior_config can only be set if with_prior is True")

    return ASRModel(
        checkpoint=checkpoint,
        network_module=network_module,
        net_args=net_args,
        prior_file=prior_file,
        prefix_name=checkpoint_name,
    )


def tune_and_evaluate_model(
        evaluation_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,

        lm_scales: List[float],
        prior_scales: List[float],

        dev_dataset_tuples: Dict[str, Any],
        vocab_opts: Dict,
        test_dataset_tuples: Optional[Dict[str, Any]] = None,
        decoder_module: str = "should_not_have_default",  # TODO: fix this - import from search instead of parameter?
        extra_forward_config: Optional[dict[str, Any]] = None,

        use_gpu: bool = False,
        debug: bool = False,

        run_test: bool = False,
) -> Tuple[Dict[str, job_path.Variable], None or GetOptimalParametersAsVariableJob]:
    """
    Example helper to execute tuning over lm_scales and prior scales.
    With the best values runs test-clean and test-other.

    This is just a reference helper and can (should) be freely changed, copied, modified etc...

    :param evaluation_name: for alias and output names
    :param asr_model: ASR model to use
    :param base_decoder_config: any decoder config dataclass
    :param lm_scales: lm scales for tuning
    :param prior_scales: prior scales for tuning, same length as lm scales
    """
    results: Dict[str, job_path.Variable] = {}

    # TUNING
    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_weight in lm_scales:
        for prior_scale in prior_scales:
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = lm_weight
            decoder_config.prior_scale = prior_scale
            search_name = f"{evaluation_name}/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)

            _, wers = search(
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
            tune_values_clean.append((wers[f"{search_name}/dev-clean"]))
            tune_values_other.append((wers[f"{search_name}/dev-other"]))
            results.update(wers)

    # EVALUATION (only if run_test)
    pick_optimal_params_job = None
    if run_test and test_dataset_tuples is not None:
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(f"{evaluation_name}/pick_best_{key}")

            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            _, wers = search(
                evaluation_name,
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


def get_best_averaged_checkpoint(base_training_name: str, train_job: ReturnnTrainingJob, num_checkpoints: int,
                                 loss_key: str) -> tuple[PtCheckpoint, str]:
    checkpoints = []
    for index in range(num_checkpoints):
        best_job = GetBestPtCheckpointJob(
            train_job.out_model_dir,
            train_job.out_learning_rates,
            key=loss_key,
            index=index,
        )
        best_job.add_alias(f"{base_training_name}/get_best_job_{index}")
        checkpoints.append(best_job.out_checkpoint)

    if num_checkpoints > 1:  # perform averaging
        avg = AverageTorchCheckpointsJob(
            checkpoints=checkpoints, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT
        )
        avg.rqmt["mem"] = 8
        return avg.out_checkpoint, f"{base_training_name}/avg_best_{num_checkpoints}_cpkt"
    elif num_checkpoints == 1:
        return checkpoints[0], f"{base_training_name}/best_cpkt"
    else:
        raise ValueError("No checkpoints found")


def get_last_averaged_checkpoint(base_training_name: str, train_job: ReturnnTrainingJob, last_n: int) -> tuple[
    PtCheckpoint, str]:
    num_checkpoints = len(train_job.out_checkpoints)
    if last_n == 0:
        return get_specific_checkpoint(base_training_name, train_job, num_checkpoints)
    avg = AverageTorchCheckpointsJob(
        checkpoints=[train_job.out_checkpoints[num_checkpoints - i] for i in range(last_n)],
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )
    return avg.out_checkpoint, f"{base_training_name}/avg_last_{num_checkpoints}_cpkt"


def get_specific_checkpoint(base_training_name: str, train_job: ReturnnTrainingJob, epoch: int) -> tuple[PtCheckpoint, str]:
    return train_job.out_checkpoints[epoch], f"{base_training_name}/ep_{epoch}_cpkt"
