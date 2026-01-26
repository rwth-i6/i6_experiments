from collections import OrderedDict
from typing import Optional, Dict, Any, Iterable

from sisyphus import job_path

from i6_core.returnn.training import (
    ReturnnTrainingJob,
    AverageTorchCheckpointsJob,
    GetBestPtCheckpointJob,
    PtCheckpoint,
)
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .asr_model import ASRModel
from .forward_job_builder import compute_ppl
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
        )

        res = calculate_perplexity(
            evaluation_name,
            asr_model,
            search_config,
            dev_dataset_tuples=dev_dataset_tuples,
            forward_method=search_config.forward_method,
            debug=search_config.debug_returnn_param,
            run_test=run_test,
            test_dataset_tuples=test_dataset_tuples,
            vocab_opts=train_data.train.target_options,
        )
        result_dict.update(res)

    return result_dict


def prepare_asr_model(
        checkpoint_name: str,
        checkpoint: PtCheckpoint,
        network_import_path: str,
        net_args,
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

    return ASRModel(
        checkpoint=checkpoint,
        network_import_path=network_import_path,
        net_args=net_args,
        prefix_name=checkpoint_name,
    )


def calculate_perplexity(
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
    calculate the perplexity on dev/test sets

    :param evaluation_name: for alias and output names
    :param asr_model: ASR model to use
    """
    results: Dict[str, job_path.Variable] = {}

    # Define params per forward_step
    if forward_method is None or forward_method == "perplexity_forward_step":
        forward_args = {}
        search_name = f"{evaluation_name}"
    else:
        raise ValueError(f"Unknown forward method: {forward_method}")

    _ = compute_ppl(
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

    # EVALUATION (only if run_test)
    if run_test and test_dataset_tuples is not None:
        for dataset in ["test-clean", "test-other"]:
            _ = compute_ppl(
                evaluation_name,
                search_config,
                asr_model=asr_model,
                forward_module=RECOGNITION_PACKAGE,
                forward_method=forward_method,
                test_dataset_tuples={dataset: test_dataset_tuples[dataset]},
                vocab_opts=vocab_opts,
                debug=debug,
                forward_args=forward_args,
                **default_returnn,
            )

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
