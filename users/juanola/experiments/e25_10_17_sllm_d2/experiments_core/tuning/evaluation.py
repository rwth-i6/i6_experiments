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
from ...constants import RECOGNITION_PACKAGE, NETWORK_MODULE
from ...default_tools import RETURNN_EXE, RETURNN_ROOT

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def create_tune_and_evaluate_jobs(
    training_name: str,
    train_job: ReturnnTrainingJob,
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

    # TODO: maybe improve
    if search_config.run_ctc_greedy_decoding:  # Run the last epoch with ctc greedy decoding
        last_epoch = max(specific_epochs)

        evaluation_name = f"{training_name}/ctc_greedy"
        checkpoint, checkpoint_name = get_specific_checkpoint(evaluation_name, train_job, last_epoch)

        # Changes
        forward_method = "forward_step_ctc_decoding"
        # Parameters could/should also be adapted

        asr_model = prepare_asr_model(
            checkpoint_name,
            checkpoint,
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
            forward_method=forward_method,
            debug=search_config.debug_returnn_param,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.dataset.target_options,
        )
        result_dict.update(res)  # ???

    return result_dict


def prepare_asr_model(
    checkpoint_name: str,
    checkpoint: PtCheckpoint,
    net_args,
    prior_args: Dict[str, Any] = None,
    prior_config: Optional[Dict[str, Any]] = None,
    datasets: Optional[TrainingDatasets] = None,
    prior_batch_size: int = 16_000,
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

    if prior_args is not None:
        returnn_config = get_prior_config(
            training_datasets=datasets,
            network_module=prior_args["network_module"],
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
        network_module=NETWORK_MODULE,
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

    v3_ctc_scale = None
    v3_lm_scale = None
    v3_prior_scale = None
    if forward_method == "forward_step_ctc_decoding":  # TODO: improve!
        v3_ctc_scale = 1.0
        v3_prior_scale = 0.0
        v3_lm_scale = 1.0
        forward_args = {
            "beam_size": search_config.beam_search.beam_size,
            "ctc_scale": v3_ctc_scale,
            "prior_scale": v3_prior_scale,
            "lm_scale": v3_lm_scale,
            #"ctc_soft_collapse_threshold": None,
            #"ctc_top_k_pruning": None,
            #"ctc_top_k_pruning_reduce_func": "mean",
        }
    else:
        forward_args = {
            "beam_size": search_config.beam_search.beam_size,
            "max_tokens_per_sec": 20,  # TODO: store somewhere
            "sample_rate": 16_000,  # TODO: get from feature extraction
        }

    # TUNING
    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_scale in search_config.lm_scales:  # todo: NOT used for now
        for prior_scale in search_config.prior_scales:
            search_name = f"{evaluation_name}/search_lm{lm_scale:.1f}_prior{prior_scale:.1f}" #TODO: improve names
            if forward_method == "forward_step_ctc_decoding": # TODO: improve!
                search_name += f"{evaluation_name}/search_lm{v3_lm_scale:.1f}_prior{v3_prior_scale:.1f}_ctc{v3_ctc_scale:.1f}"

            # OLD PARAMS:
            # "lm_weight": lm_weight,
            # "ilm_weight": None,
            # "prior_scale": prior_scale,

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

            tune_parameters.append((lm_scale, prior_scale))
            tune_values_clean.append((wers[f"{search_name}/dev-clean"]))
            tune_values_other.append((wers[f"{search_name}/dev-other"]))
            results.update(wers)

    # EVALUATION (only if run_test)
    if run_test and test_dataset_tuples is not None and forward_method != "forward_step_ctc_decoding":
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(f"{evaluation_name}/pick_best_{key}")

            # OLD PARAMS:
            # "lm_weight": pick_optimal_params_job.out_optimal_parameters[0],
            # "ilm_weight": None,
            # "prior_scale": pick_optimal_params_job.out_optimal_parameters[1],

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
