from sisyphus import Job, Task, tk


class GetTotalRuntimeFromReturnnTrainingJob(Job):
    """
    Get total train runtime
    """

    def __init__(
        self,
        returnn_learning_rates_file: tk.Path,
    ):
        self.returnn_learning_rates_file = returnn_learning_rates_file

        self.out_train_time_secs = self.output_var("out_train_time_secs.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np

        with open(self.returnn_learning_rates_file.get_path(), "rt") as file:
            scores = eval(
                file.read().strip(),
                {"EpochData": dict, "nan": float("nan"), "inf": float("inf"), "np": np},
            )
            assert isinstance(scores, dict)  # over epochs

        total_secs = sum(ep["error"][":meta:epoch_train_time_secs"] for ep in scores.values())
        self.out_train_time_secs.set(total_secs)
