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
        # Reserved for plot setting
    ):
        self.out_summary = self.output_path("summary.csv")
        self.out_plot_folder = self.output_path("plots", directory=True)
        self.out_plot = self.output_path("plots/wer_ppl.png")
        self.names = names
        self.results = results

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
        wers = [all_res["best_scores"]["test-other"] for all_res in wers]
        return ppls, wers

    def plots(self):
        import matplotlib.pyplot as plt
        import scipy.optimize
        # Apply logarithmic transformation to PPL
        ppls, wers = self.get_points()
        ln_ppl = np.log(ppls)

        # Define the regression function: WER = a + b * ln(PPL)
        def regression_func(x, a, b):
            return a + b * x

        # Fit the function to the data
        params, _ = scipy.optimize.curve_fit(regression_func, ln_ppl, wers)
        a, b = params  # Extract coefficients

        # Generate data points for the fitted line
        ppl_range = np.linspace(min(ppls), max(ppls), 100)  # Smooth range of PPL values
        ln_ppl_range = np.log(ppl_range)  # Log transform them
        wer_fit = regression_func(ln_ppl_range, a, b)  # Compute fitted WER

        # Plot Data Points
        plt.figure(figsize=(8, 6))
        markers = {"4gram":"o", "5gram":"s", "other1":"D", "other2":"v", "default":"^"}  # Different markers for different LM
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
        plt.plot(ppl_range, wer_fit, label=f"Fit: WER = {a:.2f} + {b:.2f} ln(PPL)", color="red", linestyle="--")

        # Labels and Formatting
        plt.xlabel("Perplexity (PPL)")
        plt.ylabel("Word Error Rate (WER)")
        plt.title("WER vs PPL with Log-Linear Regression")
        plt.legend()
        plt.grid(True)
        plt.xscale("log")  # Log-scale for better visualization
        plt.savefig(self.out_plot.get_path())

    def create_table(self):
        ppls, _ = self.get_points()
        import csv
        wers = list()
        for _, wer_path in self.results:
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))
        res = dict(zip(self.names, zip(ppls,wers)))
        # Define filenames
        csv_filename = self.out_summary.get_path()

        # Prepare the data as a list of lists
        table_data = [["Model Name", "Perplexity", "Best Epoch", "Dev-Clean WER", "Dev-Other WER", "Test-Clean WER",
                       "Test-Other WER"]]
        for key, values in res.items():
            ppl = values[0]
            scores = values[1]["best_scores"]
            best_epoch = values[1]["best_epoch"]
            table_data.append([key, ppl, best_epoch, scores["dev-clean"], scores["dev-other"], scores["test-clean"],
                               scores["test-other"]])

        # Save to a CSV file manually
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(table_data)
        # from matplotlib import pyplot as plt
        #
        # processor = AlignmentProcessor(
        #     alignment_bundle_path=self.alignment_bundle_path.get_path(),
        #     allophones_path=self.allophones_path.get_path(),
        #     sil_allophone=self.sil_allophone,
        #     monophone=self.monophone,
        # )
        #
        # if isinstance(self.segments, tk.Variable):
        #     segments_to_plot = self.segments.get()
        #     assert isinstance(segments_to_plot, list)
        #     out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments_to_plot]
        # elif isinstance(self.segments, Path):
        #     with open(self.segments, "rt") as segments_file:
        #         segments_to_plot = [s.strip() for s in segments_file.readlines()]
        #     out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments_to_plot]
        # else:
        #     segments_to_plot = self.segments
        #     out_plot_files = self.out_plots
        #
        # plt.rc("font", family="serif")
        #
        # for seg, out_path in zip(segments_to_plot, out_plot_files):
        #     fig, ax, *_ = processor.plot_segment(
        #         seg, font_size=self.font_size, show_labels=self.show_labels, show_title=self.show_title
        #     )
        #     if self.show_title:
        #         fig.savefig(out_path, transparent=True)
        #     else:
        #         fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
