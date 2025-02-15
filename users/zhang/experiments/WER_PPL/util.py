import glob
import numpy as np
import pickle
import os
import os.path as path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from sisyphus import tk, Job, Path, Task

import json

class WER_ppl_PlotAndSummaryJob(Job):
    # Summary the recog experiment with LMs on various ppl. Applying linear regression
    # Assumed input: A list of (ppl,WER) pairs
    # Output: A summaryfile with fitted equation, a plot image,
    __sis_hash_exclude__ = {"names": None}

    def __init__(
        self,
        names: List[str],
        results: List[Tuple[tk.Variable, tk.Path]],
        lm_scales: List[Optional[float]],
        # Reserved for plot setting
    ):
        self.out_summary = self.output_path("summary.csv")
        self.out_plot_folder = self.output_path("plots", directory=True)
        self.out_plot1 = self.output_path("plots/dev_other.png")
        self.out_plot2 = self.output_path("plots/test_other.png")
        self.names = names
        self.results = results
        self.lm_scales = lm_scales

    def tasks(self) -> Iterator[Task]:
        yield Task("create_table", mini_task=True)#, rqmt={"cpu": 1, "time": 1, "mem": 4})
        yield Task("plots", mini_task=True)

    def get_points(self):
        ppls = list()
        wers = list()
        for ppl_log, wer_path in self.results:
            """extracts ppl score from the ppl.log file"""
            with open(ppl_log.get_path(), "rt") as f:
                lines = f.readlines()[-2:]
                for line in lines:
                    line = line.split(" ")
                    for idx, ln in enumerate(line):
                        if ln == "ppl=":
                            ppls.append(float(line[idx + 1]))
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))
        ppls = ppls
        wers = {"dev-other":[all_res["best_scores"]["dev-other"] for all_res in wers], "test-other":[all_res["best_scores"]["test-other"] for all_res in wers]}
        return ppls, wers

    def plots(self):
        import matplotlib.pyplot as plt
        import scipy.optimize
        # Apply logarithmic transformation to PPL
        ppls, wers = self.get_points()
        # Define the regression function: WER = a + b * ln(PPL)
        def regression_func(x, a, b):
            return a + b * x

        def plot(ppls, wers, savepath):
            # Fit the function to the data
            ln_ppl = np.log10(ppls)
            ln_wer = np.log10(wers)
            params, _ = scipy.optimize.curve_fit(regression_func, ln_ppl, ln_wer)
            a, b = params  # Extract coefficients

            # Generate data points for the fitted line
            ppl_range = np.linspace(min(ppls), max(ppls), 100)  # Smooth range of PPL values
            ln_ppl_range = np.log10(ppl_range)  # Log transform them
            wer_fit = np.power(10, regression_func(ln_ppl_range, a, b))  # Compute fitted WER
            # Plot Data Points
            plt.figure(figsize=(8, 6))
            markers = {"4gram":"o", "5gram":"s", "2gram":"D", "3gram":"v", "default":"^"}  # Different markers for different LM
            group_points = dict([(key,[]) for key in markers])
            for i, name in enumerate(self.names):
                for key, value in markers.items():
                    if key in name:
                        group_points[key].append((ppls[i], wers[i]))
                        break

            for key, points in group_points.items():
                if len(points):
                    x_vals, y_vals = zip(*points)
                    plt.scatter(x_vals, y_vals, label=key, marker=markers[key], s=100)

            # Plot Regression Line
            plt.plot(ppl_range, wer_fit, label=f"Fit: log(WER) = {a:.2f} + {b:.2f} log(PPL)", color="red", linestyle="--")

            # Labels and Formatting
            plt.xlabel("Perplexity (PPL)")
            plt.ylabel("Word Error Rate (WER)")
            plt.title("WER vs PPL with Log-Linear Regression")
            plt.legend()
            plt.grid(True)
            plt.xscale("log")  # Log-scale for better visualization
            plt.yscale("log")
            plt.savefig(savepath)

        plot(ppls, wers["dev-other"], self.out_plot1.get_path())
        plot(ppls, wers["test-other"], self.out_plot2.get_path())

    def create_table(self):
        ppls, _ = self.get_points()
        import csv
        wers = list()
        for _, wer_path in self.results:
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))
        res = dict(zip(self.names, zip(ppls,wers,self.lm_scales)))
        # Define filenames
        csv_filename = self.out_summary.get_path()

        # Prepare the data as a list of lists
        table_data = [["Model Name", "Perplexity", "Best Epoch", "lm_scale", "Dev-Clean WER", "Dev-Other WER", "Test-Clean WER",
                       "Test-Other WER"]]
        for key, values in res.items():
            ppl = values[0]
            scores = values[1]["best_scores"]
            best_epoch = values[1]["best_epoch"]
            lm_scale = values[2]
            table_data.append([key, ppl, best_epoch, f"{lm_scale:.2f}", scores.get("dev-clean","-"), scores.get("dev-other","-"), scores.get("test-clean","-"),
                               scores.get("test-other","-")])

        # Save to a CSV file manually
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(table_data)