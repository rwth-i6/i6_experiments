from collections import OrderedDict
from typing import Optional, Dict, Any, Iterable

from sisyphus import tk, job_path

from i6_core.returnn.training import (
    ReturnnTrainingJob,
    AverageTorchCheckpointsJob,
    GetBestPtCheckpointJob,
    PtCheckpoint,
)
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .asr_model import ASRModel
from .forward_job_builder import search, compute_prior
from ..model_creation.returnn_config_helpers import get_prior_config
from ...configurations.pipeline.search_config import SearchConfig
from ...constants import RECOGNITION_PACKAGE
from ...default_tools import RETURNN_EXE, RETURNN_ROOT

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def create_tune_and_evaluate_jobs(
    training_name: str,
    train_job: ReturnnTrainingJob,
    network_import_path: str,
    net_args: dict[str, Any],
    search_config: SearchConfig,
    train_data: TrainingDatasets,
    dev_dataset_tuples: Dict[str, Any],
    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    specific_epochs: Optional[Iterable[int]] = None,
    run_best_4: bool = True,
    run_best: bool = True,
    run_test: bool = False,
    prior_args: Optional[Dict[str, Any]] = None,  # TODO: ???
) -> Dict[str, Any]:
    """
    Run evaluation jobs for different trained models

    """
    # DEFAULT PARAMETERS
    if specific_epochs is None:
        specific_epochs = train_job.returnn_config.post_config["num_epochs"]

    # Dict of all train_evals to perform (could be extended)
    checkpoint_per_evaluation = OrderedDict()
    for epoch in specific_epochs:
        evaluation_name = f"{training_name}/{epoch}"
        checkpoint_per_evaluation[evaluation_name] = get_specific_checkpoint(evaluation_name, train_job, epoch)
    if run_best_4:
        evaluation_name = f"{training_name}/best4"
        checkpoint_per_evaluation[evaluation_name] = get_best_averaged_checkpoint(
            evaluation_name, train_job, 4, search_config.avg_best_loss_name
        )
    if run_best:
        evaluation_name = f"{training_name}/best"
        checkpoint_per_evaluation[evaluation_name] = get_best_averaged_checkpoint(
            evaluation_name, train_job, 1, search_config.avg_best_loss_name
        )

    # Tune & Eval different models
    result_dict = {}
    for evaluation_name, (checkpoint, checkpoint_name) in checkpoint_per_evaluation.items():
        asr_model = prepare_asr_model(
            checkpoint_name,
            checkpoint,
            network_import_path,
            net_args,
            prior_args=prior_args,
            datasets=train_data,
            prior_batch_size=search_config.prior.batch_size,
        )

        res = tune_and_evaluate_model(
            evaluation_name,
            asr_model,
            search_config,
            dev_dataset_tuples=dev_dataset_tuples,
            forward_method=search_config.forward_method,
            debug=search_config.debug_returnn_param,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.dataset.target_options,
        )
        result_dict.update(res)

    if search_config.run_ctc_greedy_decoding_last_epoch:  # Run the last epoch with ctc greedy decoding
        res = evaluate_greedy_ctc(dev_dataset_tuples, net_args, network_import_path, prior_args, run_test,
                                  search_config, specific_epochs, test_dataset_tuples, train_data, train_job,
                                  training_name)
        result_dict.update(res)  # ???

    return result_dict


def evaluate_greedy_ctc(dev_dataset_tuples: dict[str, Any], net_args: dict[str, Any], network_import_path: str,
                        prior_args: dict[str, Any] | None, run_test: bool,
                        search_config: SearchConfig, specific_epochs: Iterable[int] | Any,
                        test_dataset_tuples: dict[str, Any] | None, train_data: TrainingDatasets,
                        train_job: ReturnnTrainingJob, training_name: str) -> dict[str, job_path.Variable]:
    results: Dict[str, job_path.Variable] = {}

    last_epoch = max(specific_epochs)
    evaluation_name = f"{training_name}/{last_epoch}_greedy_ctc"
    checkpoint, checkpoint_name = get_specific_checkpoint(evaluation_name, train_job, last_epoch)

    forward_method = "forward_step_greedy_ctc"

    asr_model = prepare_asr_model(
        checkpoint_name,
        checkpoint,
        network_import_path,
        net_args,
        prior_args=prior_args,
        datasets=train_data,
        prior_batch_size=search_config.prior.batch_size,
    )

    _, wers = search(
        evaluation_name,
        search_config,
        asr_model=asr_model,
        forward_module=RECOGNITION_PACKAGE,
        forward_method=forward_method,
        test_dataset_tuples=dev_dataset_tuples,
        debug=search_config.debug_returnn_param,
        vocab_opts=train_data.train.dataset.target_options,
        **default_returnn,
    )

    if run_test and test_dataset_tuples is not None:
        _, wers = search(
            evaluation_name,
            search_config,
            asr_model=asr_model,
            forward_module=RECOGNITION_PACKAGE,
            forward_method=forward_method,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.dataset.target_options,
            debug=search_config.debug_returnn_param,
            **default_returnn,
        )
        results.update(wers)

    return results


def prepare_asr_model(
    checkpoint_name: str,
    checkpoint: PtCheckpoint,
    network_import_path: str,
    net_args,
    prior_args: Dict[str, Any] = None,
    prior_config: Optional[Dict[str, Any]] = None,
    datasets: Optional[TrainingDatasets] = None,
    prior_batch_size: int = 16_000,
) -> ASRModel:
    """
    :param checkpoint_name:
    :param checkpoint:
    :param network_import_path:
    :param net_args:
    :param prior_args:
    :param datasets: Needed if with_prior == True
    :param prior_config: if with_prior is true, can be used to add Returnn config parameters for the prior compute job
    :return:
    """
    assert prior_args is None or datasets is not None
    assert prior_args is None or prior_config is not None

    if prior_args is not None:
        returnn_config = get_prior_config(
            training_datasets=datasets,
            network_import_path=prior_args["network_import_path"],
            config=prior_config if prior_config is not None else {},
            net_args=prior_args["net_args"],
            unhashed_net_args=prior_args.get("unhashed_net_args", None),
            debug=prior_args.get("debug", False),
            batch_size=prior_batch_size,
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
        prior_file = None
        if prior_config is not None:
            raise ValueError("prior_config can only be set if with_prior is True")

    return ASRModel(
        checkpoint=checkpoint,
        network_import_path=network_import_path,
        net_args=net_args,
        prior_file=prior_file,
        prefix_name=checkpoint_name,
    )


def tune_and_evaluate_model(
    evaluation_name: str,
    asr_model: ASRModel,
    search_config: SearchConfig,
    dev_dataset_tuples: Dict[str, Any],
    vocab_opts: Dict,
    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    forward_method: Optional[str] = None,
    debug: bool = False,
    run_test: bool = False,
) -> Dict[str, job_path.Variable]:
    """
    Helper to execute tuning over lm_scales and prior scales (over dev-clean and dev-other).
    With the best values runs test-clean and test-other.

    :param evaluation_name: for alias and output names
    :param asr_model: ASR model to use
    """
    results: Dict[str, job_path.Variable] = {}

    # Define params per forward_step

    # TUNING
    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_scale in search_config.lm_scales:
        for prior_scale in search_config.prior_scales:
            for ctc_scale in search_config.ctc_scales:
                if forward_method is None or forward_method == "forward_step":
                    forward_args = {
                        "beam_size": search_config.beam_search.beam_size,
                        "max_tokens_per_sec": 20,  # TODO: store somewhere
                        "sample_rate": 16_000,  # TODO: get from feature extraction
                    }
                    #search_name = f"{evaluation_name}/search_lm{lm_scale:.1f}_prior{prior_scale:.1f}" # OLD if hash breaks
                    search_name = f"{evaluation_name}/v1_beam{search_config.beam_search.beam_size}"
                elif forward_method == "forward_step_v2":
                    forward_args = {
                        "beam_size": search_config.beam_search.beam_size,
                        "max_tokens_per_sec": 20,  # TODO: store somewhere
                        "sample_rate": 16_000,  # TODO: get from feature extraction
                    }
                    search_name = f"{evaluation_name}/v2_beam{search_config.beam_search.beam_size}"
                elif forward_method == "forward_step_ctc_decoding":
                    forward_args = {
                        "beam_size": search_config.beam_search.beam_size,
                        "ctc_scale": ctc_scale,
                        "prior_scale": prior_scale,
                        "lm_scale": lm_scale,
                        #"ctc_soft_collapse_threshold": None,
                        #"ctc_top_k_pruning": None,
                        #"ctc_top_k_pruning_reduce_func": "mean",
                    }
                    search_name = f"{evaluation_name}/v1_beam{search_config.beam_search.beam_size}_lm{lm_scale:.1f}_prior{prior_scale:.1f}_ctc{ctc_scale:.1f}"
                else:
                    raise ValueError(f"Unknown forward method: {forward_method}")

                _, wers = search(
                    search_name,
                    search_config,
                    asr_model=asr_model,
                    forward_module=RECOGNITION_PACKAGE,
                    forward_method=forward_method,
                    test_dataset_tuples=dev_dataset_tuples,
                    debug=debug,
                    vocab_opts=vocab_opts,
                    forward_args=forward_args,
                    **default_returnn,
                )

                tune_parameters.append((lm_scale, prior_scale, ctc_scale))
                tune_values_clean.append((wers[f"{search_name}/dev-clean"]))
                tune_values_other.append((wers[f"{search_name}/dev-other"]))
                results.update(wers)

    # EVALUATION (only if run_test)
    if run_test and test_dataset_tuples is not None:
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(f"{evaluation_name}/pick_best_{key}")

            if forward_method is None or forward_method == "forward_step":
                forward_args = {
                    "beam_size": search_config.beam_search.beam_size,
                    "max_tokens_per_sec": 20,  # TODO: store somewhere
                    "sample_rate": 16_000,  # TODO: get from feature extraction
                }
            elif forward_method == "forward_step_v2":
                forward_args = {
                    "beam_size": search_config.beam_search.beam_size,
                    "max_tokens_per_sec": 20,  # TODO: store somewhere
                    "sample_rate": 16_000,  # TODO: get from feature extraction
                }
            elif forward_method == "forward_step_ctc_decoding":
                forward_args = {
                    "beam_size": search_config.beam_search.beam_size,
                    "ctc_scale": pick_optimal_params_job.out_optimal_parameters[2],
                    "prior_scale": pick_optimal_params_job.out_optimal_parameters[1],
                    "lm_scale": pick_optimal_params_job.out_optimal_parameters[0],
                    #"ctc_soft_collapse_threshold": None,
                    #"ctc_top_k_pruning": None,
                    #"ctc_top_k_pruning_reduce_func": "mean",
                }
            else:
                raise ValueError(f"Unknown forward method: {forward_method}")

            _, wers = search(
                evaluation_name,
                search_config,
                asr_model=asr_model,
                forward_module=RECOGNITION_PACKAGE,
                forward_method=forward_method,
                test_dataset_tuples={key: test_dataset_tuples[key]},
                vocab_opts=vocab_opts,
                debug=debug,
                forward_args=forward_args,
                **default_returnn,
            )
        results.update(wers)

    return results


def get_best_averaged_checkpoint(
    base_training_name: str, train_job: ReturnnTrainingJob, num_checkpoints: int, loss_key: str
) -> tuple[PtCheckpoint, str]:
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


def get_last_averaged_checkpoint(
    base_training_name: str, train_job: ReturnnTrainingJob, last_n: int
) -> tuple[PtCheckpoint, str]:
    num_checkpoints = len(train_job.out_checkpoints)
    if last_n == 0:
        return get_specific_checkpoint(base_training_name, train_job, num_checkpoints)
    avg = AverageTorchCheckpointsJob(
        checkpoints=[train_job.out_checkpoints[num_checkpoints - i] for i in range(last_n)],
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )
    return avg.out_checkpoint, f"{base_training_name}/avg_last_{num_checkpoints}_cpkt"


def get_specific_checkpoint(
    base_training_name: str, train_job: ReturnnTrainingJob, epoch: int
) -> tuple[PtCheckpoint, str]:
    return train_job.out_checkpoints[epoch], f"{base_training_name}/ep_{epoch}_cpkt"
