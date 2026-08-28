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

        # Fail rather than report a partial sum or nan: the output would be cached under the
        # same hash, so a later rerun of the training with the info would not update it.
        key = ":meta:epoch_train_time_secs"
        missing = [ep for ep, d in scores.items() if key not in d["error"]]
        if missing:
            raise Exception(
                f"{self}: {len(missing)} of {len(scores)} epochs without {key}"
                f" in {self.returnn_learning_rates_file}."
                " The training predates the engines writing it; rerun the training for the runtime."
            )
        self.out_train_time_secs.set(sum(d["error"][key] for d in scores.values()))
