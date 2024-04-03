"""
Given some training job, get the training time per epoch
"""

from typing import Union, Any, Dict, Set, List
import re
from sisyphus import Job
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.users.zeyer.utils.job_dir import get_job_base_dir


def get_training_times_per_epoch(
    training_job: Union[str, ReturnnTrainingJob],
    *,
    expected_gpu: str,
    ignore_first_n_epochs: int = 0,
) -> List[Union[float, int]]:
    """
    :param training_job: reads out_learning_rates and the log from it
    :param expected_gpu: e.g. "NVIDIA GeForce GTX 1080 Ti"
    :param ignore_first_n_epochs: ignore the first N epochs,
        e.g. because we might use smaller models, or data filtering, or so,
        so they are much faster and not comparable to the others
    :return: avg time per epoch in secs
    """
    # We can read the learning_rates to get the time per epoch in secs.
    job_dir = get_job_base_dir(training_job)
    scores = _read_scores_and_learning_rates(f"{job_dir}/output/learning_rates")
    epoch_times = _read_epoch_times_from_scores_and_learning_rates(scores)
    epoch_times = {ep: t for ep, t in epoch_times.items() if ep > ignore_first_n_epochs}
    epoch_steps = _read_epoch_steps_from_scores_and_learning_rates(scores)
    epoch_steps = {ep: t for ep, t in epoch_steps.items() if ep > ignore_first_n_epochs}
    epoch_steps_min = min(epoch_steps.values())
    epoch_steps_max = max(epoch_steps.values())
    assert epoch_steps_max - epoch_steps_min <= epoch_steps_max * 0.1, f"epoch_steps: {epoch_steps}"  # sanity check

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
    from i6_experiments.users.zeyer.utils.job_log import open_job_logs

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("job", help="job dir")
    arg_parser.add_argument("--gpu", required=True, help="expected GPU, e.g. 'NVIDIA GeForce GTX 1080 Ti'")
    arg_parser.add_argument("--ignore-first-n-epochs", type=int, default=0)
    args = arg_parser.parse_args()

    times_per_epoch = get_training_times_per_epoch(
        args.job, expected_gpu=args.gpu, ignore_first_n_epochs=args.ignore_first_n_epochs
    )
    print("times per epoch:")
    print(f"(num epochs: {len(times_per_epoch)})")
    min_, max_, avg = min(times_per_epoch), max(times_per_epoch), sum(times_per_epoch) / len(times_per_epoch)
    print(f"min, max, avg: {min_:.2f}, {max_:.2f}, {avg:.2f}")
    times_per_epoch.sort()
    print(f"median: {times_per_epoch[len(times_per_epoch) // 2]:.2f}")

    for log_file, log_filename in open_job_logs(args.job):
        print("log file:", log_filename)
        for i, line in enumerate(log_file):
            if line.startswith("Host: ") or line.startswith("Load: "):
                print(line.rstrip())
            if i > 100:
                break


if __name__ == "__main__":
    main()
