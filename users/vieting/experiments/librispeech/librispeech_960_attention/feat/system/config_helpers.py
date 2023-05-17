"""
Helper functions that are used in config and more specific than the general pipeline helpers.
"""
import os.path

from i6_experiments.users.vieting.tools.report import Report
from .attention_asr_config import create_config
from .data import (
    build_training_datasets,
    build_test_dataset,
)
from ..default_tools import (
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
)
from .feature_extraction_net import (
    log10_net_10ms,
)
from .pipeline import (
    training,
    search,
    get_average_checkpoint,
    get_best_checkpoint,
)


def get_test_dataset_tuples(bpe_size):
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            testset,
            use_raw_features=True,
            bpe_size=bpe_size,
        )
    return test_dataset_tuples


def run_train(
        exp_name,
        train_args,
        train_data,
        feature_extraction_net,
        feature_extraction_name,
        num_epochs,
        recog_epochs,
        prefix_name="",
        **kwargs,
):
    exp_prefix = os.path.join(prefix_name, exp_name)
    returnn_config = create_config(
        training_datasets=train_data,
        **train_args,
        feature_extraction_net=feature_extraction_net,
        feature_extraction_name=feature_extraction_name,
        recog_epochs=recog_epochs,
    )
    train_job = training(
        exp_prefix,
        returnn_config,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
        num_epochs=num_epochs,
        gpu_mem=kwargs.get("gpu_mem", 11),
    )
    return train_job


def run_search(
        exp_name,
        train_args,
        train_data,
        train_job,
        feature_extraction_net,
        feature_extraction_name,
        num_epochs,
        search_args,
        recog_epochs,
        bpe_size,
        prefix_name="",
        **kwargs,
):
    exp_prefix = os.path.join(prefix_name, exp_name)

    search_args = search_args if search_args is not None else train_args

    returnn_search_config = create_config(
        training_datasets=train_data,
        **search_args,
        feature_extraction_net=feature_extraction_net,
        feature_extraction_name=feature_extraction_name,
        is_recog=True,
    )

    num_avg = kwargs.get("num_avg", 4)
    averaged_checkpoint = get_average_checkpoint(
        train_job,
        returnn_exe=RETURNN_CPU_EXE,
        returnn_root=RETURNN_ROOT,
        num_average=num_avg,
    )

    best_checkpoint = get_best_checkpoint(train_job)

    if recog_epochs is None:
        if num_epochs <= 100:
            default_recog_epochs = [20, 40]
        else:
            default_recog_epochs = []
        default_recog_epochs += [80 * i for i in range(1, int(num_epochs / 80) + 1)]
        if num_epochs % 80 != 0:
            default_recog_epochs += [num_epochs]
    else:
        default_recog_epochs = recog_epochs

    test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

    for ep in default_recog_epochs:
        search(
            exp_prefix + f"/recogs/ep-{ep}",
            returnn_search_config,
            train_job.out_checkpoints[ep],
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

    wers = {}
    wers["last"] = search(
        exp_prefix + "/default_last",
        returnn_search_config,
        train_job.out_checkpoints[num_epochs],
        test_dataset_tuples,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
    )

    wers["best"] = search(
        exp_prefix + "/default_best",
        returnn_search_config,
        best_checkpoint,
        test_dataset_tuples,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
    )

    wers["average"] = search(
        exp_prefix + f"/average_{num_avg}",
        returnn_search_config,
        averaged_checkpoint,
        test_dataset_tuples,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
    )
    return wers


def run_exp(
        exp_name,
        train_args,
        feature_extraction_net=log10_net_10ms,
        feature_extraction_name="log_mel_features",
        num_epochs=300,
        search_args=None,
        recog_epochs=None,
        bpe_size=10000,
        **kwargs,
):
    report = Report(columns_start=["name"], columns_end=["dev-clean", "dev-other", "test-clean", "test-other"])
    report_args = {"name": exp_name}
    if train_args.get("retrain_checkpoint", None):
        assert kwargs.get("epoch_wise_filter", None) is None, "epoch_wise_filter should be disabled for retraining."
    train_data = build_training_datasets(
        bpe_size=bpe_size,
        use_raw_features=True,
        epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
        link_speed_perturbation=train_args.get("speed_pert", True),
        seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
    )
    train_job = run_train(
        exp_name,
        train_args,
        train_data,
        feature_extraction_net,
        feature_extraction_name,
        num_epochs,
        recog_epochs,
        **kwargs,
    )

    wers = run_search(
        exp_name,
        train_args,
        train_data,
        train_job,
        feature_extraction_net,
        feature_extraction_name,
        num_epochs,
        search_args,
        recog_epochs,
        bpe_size=bpe_size,
        **kwargs,
    )

    for checkpoint in wers:
        report.add({**report_args, "checkpoint": checkpoint, **wers[checkpoint]})
    return train_job, train_data, report
