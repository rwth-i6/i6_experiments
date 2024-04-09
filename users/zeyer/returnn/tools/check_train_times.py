"""
Given some training job, get the training time per epoch

Call this via:

    export PYTHONPATH=recipe
    python3 -m i6_experiments.users.zeyer.returnn.tools.check_train_times ...
"""

from typing import Optional, Union, Any, Dict, Set, List, Tuple
import os
import re
import numpy as np
from sisyphus import Job
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.users.zeyer.utils.job_dir import get_job_base_dir


def get_training_times_per_epoch(
    training_job: Union[str, ReturnnTrainingJob],
    *,
    expected_gpu: str,
    ignore_first_n_epochs: int = 0,
    min_required: Optional[int] = None,
) -> List[Union[float, int]]:
    """
    :param training_job: reads out_learning_rates and the log from it
    :param expected_gpu: e.g. "NVIDIA GeForce GTX 1080 Ti"
    :param ignore_first_n_epochs: ignore the first N epochs,
        e.g. because we might use smaller models, or data filtering, or so,
        so they are much faster and not comparable to the others
    :param min_required:
    :return: avg time per epoch in secs
    """
    # We can read the learning_rates to get the time per epoch in secs.
    job_dir = get_job_base_dir(training_job)
    scores_and_lr_filename = f"{job_dir}/work/learning_rates"
    if not os.path.exists(scores_and_lr_filename):
        scores_and_lr_filename = f"{job_dir}/output/learning_rates"
    scores = _read_scores_and_learning_rates(scores_and_lr_filename)
    scores = {epoch: v for epoch, v in scores.items() if epoch > ignore_first_n_epochs}
    scores, filtered_by_device = _filter_learning_rates_by_device(
        scores, device=expected_gpu, min_required=min_required or len(scores)
    )
    epoch_times = _read_epoch_times_from_scores_and_learning_rates(scores)
    epoch_times = {ep: t for ep, t in epoch_times.items()}
    epoch_steps = _read_epoch_steps_from_scores_and_learning_rates(scores)
    epoch_steps = {ep: t for ep, t in epoch_steps.items()}
    epoch_steps_min = min(epoch_steps.values())
    epoch_steps_max = max(epoch_steps.values())
    assert epoch_steps_max - epoch_steps_min <= epoch_steps_max * 0.1, f"epoch_steps: {epoch_steps}"  # sanity check

    if not filtered_by_device:
        # We also need to check that we have the same GPU. For that, we currently need to check the log.
        gpus = _read_used_gpus_from_log(job_dir)
        assert gpus == {expected_gpu}, f"expected GPU {expected_gpu}, found in log: {gpus}"

    return list(epoch_times.values())


def _read_scores_and_learning_rates(filename: str) -> Dict[int, Dict[str, Union[float, Any]]]:
    # simple wrapper, to eval newbob.data
    # noinspection PyPep8Naming
    def EpochData(learningRate: float, error: Dict[str, Union[float, Any]]) -> Dict[str, Union[float, Any]]:
        """
        :param learningRate:
        :param error: keys are e.g. "dev_score_output" etc
        """
        assert isinstance(error, dict)
        d = {"learning_rate": learningRate}
        d.update(error)
        return d

    # nan/inf, for some broken newbob.data
    nan = float("nan")
    inf = float("inf")

    scores_str = open(filename).read()
    scores = eval(scores_str, {"EpochData": EpochData, "nan": nan, "inf": inf})
    assert isinstance(scores, dict)
    return scores


def _read_epoch_times_from_scores_and_learning_rates(scores: Dict[int, Dict[str, Any]]) -> Dict[int, Union[float, int]]:
    return {epoch: v[":meta:epoch_train_time_secs"] for epoch, v in scores.items()}


def _read_epoch_steps_from_scores_and_learning_rates(scores: Dict[int, Dict[str, Any]]) -> Dict[int, Union[float, int]]:
    return {epoch: v[":meta:epoch_num_train_steps"] for epoch, v in scores.items()}


def _filter_learning_rates_by_device(
    scores: Dict[int, Dict[str, Any]], *, device: str, min_required: int
) -> Tuple[Dict[int, Dict[str, Any]], bool]:
    """
    :return: filtered scores if we can apply the filter, whether we applied the filter
    """
    assert len(scores) >= min_required
    res = {epoch: v for epoch, v in scores.items() if v.get(":meta:device") == device}
    if len(res) < min_required:
        return scores, False
    return res, True


def _read_used_gpus_from_log(job: Union[str, Job]) -> Set[str]:
    from i6_experiments.users.zeyer.utils.job_log import open_job_logs

    gpus = set()

    for log_file, log_filename in open_job_logs(job):
        for line in log_file:
            # Example: "Using gpu device 3: NVIDIA GeForce GTX 1080 Ti"
            if not line.startswith("Using gpu device "):
                continue
            m = re.match(r"^Using gpu device (\d+): (.*)$", line)
            gpu = m.group(2)
            gpus.add(gpu)

    assert gpus
    return gpus


def main():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("job", help="job dir")
    arg_parser.add_argument("--gpu", required=True, help="expected GPU, e.g. 'NVIDIA GeForce GTX 1080 Ti'")
    arg_parser.add_argument("--ignore-first-n-epochs", type=int, default=0)
    arg_parser.add_argument("--take-n-fastest-epochs", type=int, default=None, help="take the N fastest epochs")
    args = arg_parser.parse_args()

    times_per_epoch = get_training_times_per_epoch(
        args.job,
        expected_gpu=args.gpu,
        ignore_first_n_epochs=args.ignore_first_n_epochs,
        min_required=args.take_n_fastest_epochs,
    )
    print("times per epoch:")
    print(f"(num epochs: {len(times_per_epoch)})")
    print(f"min, max: {min(times_per_epoch):.2f}, {max(times_per_epoch):.2f}")
    print(f"mean: {np.mean(times_per_epoch):.2f}, std: {np.std(times_per_epoch):.2f}")
    times_per_epoch.sort()
    print(f"median: {times_per_epoch[len(times_per_epoch) // 2]:.2f}")
    if args.take_n_fastest_epochs:
        assert len(times_per_epoch) >= args.take_n_fastest_epochs, f"only {len(times_per_epoch)} epochs"
        times_per_epoch = times_per_epoch[: args.take_n_fastest_epochs]
    print(f"after outlier removal: (num: {len(times_per_epoch)})")
    print(f"min, max: {min(times_per_epoch):.2f}, {max(times_per_epoch):.2f}")
    print(f"mean: {np.mean(times_per_epoch):.2f}, std: {np.std(times_per_epoch):.2f}")
    print(f"median: {times_per_epoch[len(times_per_epoch) // 2]:.2f}")


def _z_score_outlier_removal(ls: List[float], threshold: float = 3.0) -> List[float]:
    ls = np.array(ls)
    mean = np.mean(ls)
    std = np.std(ls)
    z_scores = (ls - mean) / std
    return [v for v, z in zip(ls, z_scores) if abs(z) <= threshold]


def _zmin_score_outlier_removal(ls: List[float], threshold: float = 3.0) -> List[float]:
    ls = np.array(ls)
    min_ = np.min(ls)
    std = np.std(ls)
    z_scores = (ls - min_) / std
    return [v for v, z in zip(ls, z_scores) if abs(z) <= threshold]


def _debug_out_job_logs(job: str):
    from i6_experiments.users.zeyer.utils.job_log import open_job_logs

    for log_file, log_filename in open_job_logs(job):
        print("log file:", log_filename)
        for i, line in enumerate(log_file):
            if line.startswith("Host: ") or line.startswith("Load: "):
                print(line.rstrip())
            if i > 100:
                break


if __name__ == "__main__":
    main()
