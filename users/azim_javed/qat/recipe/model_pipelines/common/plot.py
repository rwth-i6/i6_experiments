from typing import List, Optional

import numpy as np
from sisyphus import Job, Task, tk


class PlotTrainingJob(Job):
    def __init__(
        self,
        learning_rates: tk.Path,
        metrics: Optional[List[str]] = None,
        title: Optional[str] = None,
        log_scale: bool = False,
    ):
        self.learning_rates = learning_rates
        self.metrics = metrics
        self.title = title
        self.log_scale = log_scale
        self.out_plot_se = self.output_path("score_and_error.png")
        self.out_plot_lr = self.output_path("learning_rate.png")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        with open(self.learning_rates.get_path()) as learning_rate_file:
            data = eval(
                learning_rate_file.read(),
                {"__builtins__": {}, "EpochData": EpochData, "nan": float("nan"), "inf": float("inf"), "np": np},
            )
        epochs = sorted(data)
        if not epochs:
            raise ValueError("No epochs in learning_rates")
        available_metrics = {key for epoch in epochs for key in data[epoch]["error"]}
        metrics = sorted(available_metrics) if self.metrics is None else self.metrics
        if not metrics:
            raise ValueError("No metrics to plot")
        missing = set(metrics) - available_metrics
        if missing:
            raise ValueError(f"Unknown metrics: {sorted(missing)!r}; available: {sorted(available_metrics)!r}")

        fig, ax = plt.subplots()
        for metric in metrics:
            values = [data[epoch]["error"].get(metric, np.nan) for epoch in epochs]
            ax.plot(epochs, values, label=metric)
        if self.log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("subepoch")
        ax.set_ylabel("score / loss / error")
        ax.set_title(self.title or "Training metrics")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.out_plot_se.get_path())
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.semilogy(epochs, [data[epoch]["learning_rate"] for epoch in epochs])
        ax.set_xlabel("subepoch")
        ax.set_ylabel("learning rate")
        ax.set_title(self.title or "Learning rate")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.out_plot_lr.get_path())
        plt.close(fig)
