import glob
import numpy as np
import pickle
import os
import os.path as path
import subprocess as sp
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

        lm_tunes: List[Optional[tk.Path]],
        prior_tunes: List[Optional[tk.Path]],
        search_errors: List[Optional[tk.Path]],
        search_errors_rescore: List[Optional[tk.Path]],
        lm_default_scales: List[Optional[float]],
        prior_default_scales: List[Optional[float]],
        eval_dataset_keys: List[str] = ["test-other","dev-other"],
        # Reserved for plot setting
    ):
        self.out_summary = self.output_path("summary.csv")
        self.out_plot_folder = self.output_path("plots", directory=True)
        # self.out_plot1 = self.output_path("plots/dev_other.png")
        # self.out_plot2 = self.output_path("plots/test_other.png")
        # self.out_plot3 = self.output_path("plots/dev_clean.png")
        # self.out_plot4 = self.output_path("plots/test_clean.png")
        self.out_plots = [self.output_path(f"plots/{key}.png") for key in eval_dataset_keys]
        self.names = names
        self.results = results
        self.lm_tunes = lm_tunes
        self.prior_tunes = prior_tunes
        self.lm_default_scales = lm_default_scales
        self.prior_default_scales = prior_default_scales
        self.search_errors = search_errors
        self.search_errors_rescore = search_errors_rescore
        self.eval_dataset_keys = eval_dataset_keys

    def tasks(self) -> Iterator[Task]:
        yield Task("create_table", mini_task=True)#, rqmt={"cpu": 1, "time": 1, "mem": 4})
        yield Task("plots", mini_task=True)

    def get_points(self):
        ppls = list()
        wers = list()
        for ppl_log, wer_path in self.results:
            """extracts ppl score from the ppl.log file"""
            if isinstance(ppl_log, float):
                ppls.append(ppl_log)
            else:
                with open(ppl_log.get_path(), "rt") as f:
                    lines = f.readlines()[-2:]
                    for line in lines:
                        line = line.split(" ")
                        for idx, ln in enumerate(line):
                            if ln == "ppl=" or ln == "Perplexity:":
                                ppls.append(float(line[idx + 1]))
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))
        ppls = ppls
        #wers = {"dev-other":[all_res["best_scores"]["dev-other"] for all_res in wers], "test-other":[all_res["best_scores"]["test-other"] for all_res in wers]}
        wers = {key: [all_res["best_scores"][key] for all_res in wers] for key in self.eval_dataset_keys}

        return ppls, wers

    def plots(self):
        import matplotlib.pyplot as plt
        import scipy.optimize
        # Apply logarithmic transformation to PPL
        ppls, wers = self.get_points()
        # Define the regression function: WER = a + b * ln(PPL)
        def regression_func(x, a, b):
            return a + b * x

        def plot(ppls, wers, savepath, names):
            import matplotlib.ticker as mticker
            # Fit the function to the data
            ln_ppl = np.log10(ppls)
            ln_wer = np.log10(wers)
            params, _ = scipy.optimize.curve_fit(regression_func, ln_ppl, ln_wer)
            a, b = params  # Extract coefficients

            # Generate data points for the fitted line
            ppl_range = np.linspace(min(ppls), max(ppls), 100)
            ln_ppl_range = np.log10(ppl_range)
            wer_fit = np.power(10, regression_func(ln_ppl_range, a, b))

            # Plot Data Points
            fig, ax = plt.subplots(figsize=(8, 6))
            markers = {"trafo": "D", "Llama": "v", "gram": "o", "ffnn": "s", "uniform": "^"} #"2gram": "+"
            group_points = {key: [] for key in markers}
            for i, name in enumerate(names):
                for key, m in markers.items():
                    if key in name:
                        group_points[key].append((ppls[i], wers[i]))
                        break

            for key, points in group_points.items():
                if points:
                    x_vals, y_vals = zip(*points)
                    ax.scatter(x_vals, y_vals, label=key, marker=markers[key], s=100)

            # Plot Regression Line
            ax.plot(ppl_range, wer_fit,
                    label=f"Fit: log(WER) = {a:.2f} + {b:.2f} log(PPL)",
                    color="red", linestyle="--")

            # Labels and Formatting
            ax.set_xlabel("Perplexity (PPL)")
            ax.set_ylabel("Word Error Rate (WER)")
            ax.set_title("WER vs PPL with Log-Linear Regression")
            ax.legend()
            ax.grid(True)
            ax.set_xscale("log")
            ax.set_yscale("log")

            # --- THIS IS THE KEY PART: TURN OFF SCIENTIFIC NOTATION ---
            for axis in (ax.xaxis, ax.yaxis):
                fmt = mticker.ScalarFormatter()
                fmt.set_scientific(False)  # turn off 1e3 offset notation
                fmt.set_useOffset(False)  # turn off “offset” printing
                axis.set_major_formatter(fmt)

            plt.tight_layout()
            plt.savefig(savepath)
            plt.close(fig)

        for i, key in enumerate(self.eval_dataset_keys):
            plot(ppls, wers[key], self.out_plots[i].get_path(),self.names)
        # plot(ppls, wers["dev-other"], self.out_plot1.get_path(),self.names)
        # plot(ppls, wers["test-other"], self.out_plot2.get_path(),self.names)
        # plot(ppls, wers["dev-clean"], self.out_plot3.get_path(), self.names)
        # plot(ppls, wers["test-clean"], self.out_plot4.get_path(), self.names)


    def create_table(self):
        ppls, _ = self.get_points()
        import csv
        wers = list()
        for _, wer_path in self.results:
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))
        res = dict(zip(self.names, zip(ppls,wers,self.lm_tunes,self.search_errors,self.search_errors_rescore, self.lm_default_scales,self.prior_tunes,self.prior_default_scales)))
        # Define filenames
        csv_filename = self.out_summary.get_path()

        def retrieve_from_file(file, default_scale): # This case we use offset as tune value
            lm_weight_tune = json.load(open(file))
            lm_weight_tune = lm_weight_tune["best_tune"]
            return default_scale + lm_weight_tune

        # Prepare the data as a list of lists
        dataset_header = [dataset_key + " WER" for dataset_key in self.eval_dataset_keys]
        table_data = [["Model Name", "Perplexity", "lm_scale", "prior_scale", "search_error", "search_error_rescore"] + dataset_header]
        for key, values in res.items():
            ppl = values[0]
            scores = values[1]["best_scores"]
            best_lm_tune = values[2]
            if best_lm_tune:
                if isinstance(best_lm_tune, tk.Variable):
                    lm_scale = best_lm_tune.get()
                else: # Warning: assume best_lm_tune to be a path and an offset
                    lm_scale = retrieve_from_file(best_lm_tune, values[5])
            else:
                lm_scale = values[5]
            # if best_lm_tune and os.path.exists(best_lm_tune):
            #     lm_weight_tune = json.load(open(best_lm_tune))
            #     lm_weight_tune = lm_weight_tune["best_tune"]
            #     lm_scale = values[4] + lm_weight_tune
            # else:
            #     lm_scale = values[4]
            best_prior_tune = values[6]
            if best_prior_tune:
                if isinstance(best_prior_tune, tk.Variable):
                    prior_scale = best_prior_tune.get()
                else: # Warning: assume best_prior_tune to be a path and an offset
                    prior_scale = retrieve_from_file(best_prior_tune, values[7])
            else:
                prior_scale = values[7]

            # In default, if lm_scale is 0, we will set prior scale also to 0 for search
            # if lm_scale == 0:
            #     prior_scale = 0

            search_error = "-"
            if values[3]:
                with open(values[3].get_path()) as f:
                    import re
                    search_error = f.readline()
                    search_error = re.search(r"([-+]?\d*\.\d+|\d+)%", search_error).group(0)
            search_error_rescore = "-"
            if values[4]:
                with open(values[4].get_path()) as f:
                    import re
                    search_error_rescore = f.readline()
                    search_error_rescore = re.search(r"([-+]?\d*\.\d+|\d+)%", search_error_rescore).group(0)
            row = [key, f"{ppl:.2f}", f"{lm_scale:.2f}",f"{prior_scale:.2f}", search_error, search_error_rescore]
            for dataset_key in self.eval_dataset_keys:
                row.append(scores.get(dataset_key,"-"))
            table_data.append(row)

        # Save to a CSV file manually
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(table_data)

class GnuPlotJob(Job):
    def __init__(
        self,
        res_table: tk.Path,
        eval_dataset_keys: List[str] = ["test-other","dev-other"],
        curve_point: int = 0,
        version: int = 2,
    ):
        self.input_summary = res_table
        self.curve_point = curve_point
        self.out_plot_dir = self.output_path("plots", directory=True)
        self.out_equations_dir = self.output_path("regressions", directory=True)
        self.out_equations = dict(zip(eval_dataset_keys,[self.output_path(f"regressions/{key}.txt") for key in eval_dataset_keys]))
        self.out_scripts = dict(zip(eval_dataset_keys,[self.output_path(f"wer_vs_ppl_{key}.gb") for key in eval_dataset_keys]))
        self.out_plots = dict(zip(eval_dataset_keys,[self.output_path(f"plots/{key}.pdf") for key in eval_dataset_keys]))
        self.eval_dataset_keys = eval_dataset_keys

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)#, rqmt={"cpu": 1, "time": 1, "mem": 4})

    def split_dat(self):
        from collections import defaultdict
        import os

        def get_idx_name(header:List[str], dataset_key: str):
            for name in header:
                if dataset_key.lower() in name.lower():
                    return name
            raise Exception(f"dataset key {dataset_key} not found.")
        os.makedirs(self.out_plot_dir.get_path(), exist_ok=True)
        # read header and locate columns

        with open(self.input_summary.get_path(), 'r') as f:
            header = next(f).strip().split(',')
            all_lines = list(f)  # Read all remaining lines into memory

        for data_setkey in self.eval_dataset_keys:
            groups = defaultdict(list)  # <-- reset for each dataset
            idx_wer = header.index(get_idx_name(header, data_setkey))
            idx_name = header.index('Model Name')
            idx_ppl = header.index('Perplexity')

            for line in all_lines:
                cols = line.strip().split(',')
                lm = cols[idx_name].split('_')[0]
                if "gram" in lm:
                    lm = "Count-based"
                elif "trafo" in lm:
                    lm = "Transformer"
                ppl = float(cols[idx_ppl])
                wer = float(cols[idx_wer])
                groups[lm].append((ppl, wer))

            for lm, pts in groups.items():
                out_file = self.out_plot_dir.get_path() + f"/{lm}_{data_setkey}.dat"
                with open(out_file, 'w') as o:
                    for ppl, wer in sorted(pts):
                        o.write(f"{ppl}\t{wer}\n")

    def merge_data(self, datafiles: List[str], data_setkey: str):
        merged_path = os.path.join(self.out_plot_dir.get_path(), f"all_{data_setkey}.dat")
        labels = []
        with open(merged_path, "w") as fout:
            for fn in datafiles:
                # derive prefix: e.g. "plots/ffnn8_test-other.dat" → "ffnn8"
                prefix = os.path.basename(fn).rsplit(f"_{data_setkey}.dat", 1)[0]
                if "ffnn" in prefix:
                    prefix = "FeedForward"
                elif "gram" in prefix:
                    prefix = "Ngram"
                labels.append(prefix)
                with open(fn) as fin:
                    fout.write(fin.read())
        return merged_path, " ".join(labels)

    def make_tics_string(self, all_dat: str,
                         axis: int = 0,
                         num_ticks: int = 6,
                         pad_frac: float = 0.05):
        """
        all_dat   : path to whitespace‐delimited two‐column PPL/WER file
        axis      : 0 for x (PPL), 1 for y (WER)
        num_ticks : how many ticks to generate
        pad_frac  : fraction to pad below/above the data range
        returns   : e.g. '("10" 10, "20" 20, "30" 30, "40" 40)'
        """
        import numpy as np
        import math
        # 1) load
        data = np.loadtxt(all_dat)
        vals = data[:, axis]
        lo, hi = vals.min(), vals.max()

        # 2) pad
        span = hi - lo
        lo = max(lo - span * pad_frac, 1.0)
        hi = hi + span * pad_frac

        # 2) work in log10 space
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        span = log_hi - log_lo

        # 3) pad the log range
        log_lo -= pad_frac * span
        log_hi += pad_frac * span

        # 4) linearly spaced exponents
        exps = np.linspace(log_lo, log_hi, num_ticks)
        if self.curve_point:
            exps = np.append(exps, math.log10(self.curve_point))
        # 5) back-transform and unique/round
        ticks = sorted(set(10 ** exps))

        # 6) format for Gnuplot: each -> '"label" pos'
        if vals.max()<10:
            entries = [f'"{vals.min():.1f}" {vals.min():.1f}']
        else:
            entries = [f'"{vals.min():.0f}" {vals.min():.0f}']

        for t in ticks:
            # choose a sensible label formatting:
            if self.curve_point:
                dif = abs(math.log10(t) - math.log10(self.curve_point))
                if dif < 0.08 and dif > 1e-3:
                    continue
            if t < 1:
                label = f"{t:.2f}"
            elif t < 10:
                label = f"{t:.1f}"
            else:
                if axis == 1:
                    label = f"{t:.1f}"
                else:
                    label = f"{t:.0f}"
            entries.append(f'"{label}" {t:g}')
        if vals.max()<10:
            entries.append(f'"{vals.max():.1f}" {vals.max():.1f}')
        else:
            if axis == 1:
                entries.append(f'"{vals.max():.1f}" {vals.max():.1f}')
            else:
                entries.append(f'"{vals.max():.0f}" {vals.max():.0f}')
        range = f"[{lo}:{hi}]"
        return "(" + ", ".join(entries) + ")", range

    def plot_gnuplot(self, dataset_key: str):

        import textwrap
        # build gnuplot script dynamically
        plt_path = self.out_scripts[dataset_key].get_path()
        dat_pattern = os.path.join(self.out_plot_dir.get_path(), f"*_{dataset_key}.dat")
        data_files  = sorted(glob.glob(dat_pattern))

        if not data_files:
            raise RuntimeError(f"No data files match {dat_pattern!r}, can't plot WER vs PPL")

        # 2) turn them into a single space-separated string,
        #    each path quoted if it has spaces
        def q(path):
            return f'"{path}"' if " " in path else path

        files_str = " ".join(q(p) for p in data_files)
        merged_path, labels = self.merge_data(data_files, dataset_key)

        xtics, xrange = self.make_tics_string(merged_path, axis=0, num_ticks=5, pad_frac=0.01)
        ytics, yrange = self.make_tics_string(merged_path, axis=1, num_ticks=6)
        # Gnuplot script
        gp = textwrap.dedent(f"""
            #standard setup
            set terminal pdf size 6,4
            set output "{self.out_plots[dataset_key].get_path()}"
            
            set key outside right center box
            set border linewidth 2
            set tics scale 1.5
            set xlabel "Perplexity (PPL)"
            set ylabel "Word Error Rate (%)"
            set grid
            # Log scales on both axes
            set logscale x 10
            set logscale y 10
            
            #model + back-transform functions
            f(x)     = a * x + b
            f_log(x) = a * log10(x) + b
            f_real(x)= 10**(f_log(x))
            
            #Optional fit on saturated part
            f1(x) = a1 * x + b1
            f1_log(x) = a1 * log10(x) + b1
            f1_real(x) = 10**(f1_log(x))
            
            #single fit on merged data
            fit {f'[log10(5):log10({self.curve_point})]' if self.curve_point else ''} f(x) "{merged_path}" using (log10($1)):(log10($2)) via a,b
            {''if self.curve_point else '#'}fit [log10({self.curve_point}):log10(185)] f1(x) "{merged_path}" using (log10($1)):(log10($2)) via a1, b1

            #Save the fit equation
            set print "{self.out_equations[dataset_key].get_path()}"
            print sprintf("eq1: log(WER) = %.2f + %.2f * log(PPL)", b, a)
            {'print sprintf("eq2: log(WER) = %.2f + %.2f * log(PPL)", b1, a1)' if self.curve_point else ''}
            set print      # close the print destination back to the console
            
            #Find true WER range from the merged file
            # stats "{merged_path}" using 2 name "Y" nooutput
            # Y_min and Y_max now exist
            
            # ---Set y axis
            #Pad it by, say, 5% above & below (in linear WER space!)
            # pad = 0.05
            # y_lo = Y_min * (1 - pad)
            # y_hi = Y_max * (1 + pad)
            set yrange {yrange}
            set ytics {ytics}#("2.0" 2.0, "4.0" 4.0, "6.0" 6.0, "8.0" 8.0, "10.0" 10.0, "12.0" 12.0, "14.0" 14.0, "16.0" 16.0)
            
            #–– linear‐space Y-tics at 10% steps:
            # dy = 10
            # set ytics y_lo, dy, y_hi
            
            # ---X–axis similarly
            # stats "{merged_path}" using 1 name "X" nooutput
            # x_lo = X_min * 0.9
            # x_hi = X_max * 1.1
            set xrange {xrange}
            set xtics {xtics}#("10" 10, "20" 20, "30" 30, "40" 40, "50" 50, "60" 60, "70" 70, "80" 80, "90" 90, "100" 100, "185" 185)
            # dx = 10
            # set xtics x_lo, dx, X_max
            # # format the label
            # label_hi = sprintf("%g", X_max)
            # 
            # set xtics add ( label_hi X_max )
            
            #Define point-types & colors
            set style line 1 lt 1 lc rgb "red" pt 1 ps 0.8  # cross
            set style line 2 lt 1 lc rgb "#006400" pt 8 ps 0.8  # triangle 
            set style line 3 lt 1 lc rgb "black" pt 4 ps 0.8  # square 
            set style line 4 lt 1 lc rgb "red" pt 1 ps 0.8 # triangle
            set style line 5 lt 1 lc rgb "blue" pt 11 ps 0.8 # cross
            
            # regression line style
            set style line 6 lt 2 lc rgb "blue" lw 2 dashtype 3
            # set style line 7 lt 2 lc rgb "blue" lw 2 dashtype 3
            set style line 7 lt 2 lc rgb "black" lw 2 dashtype 3
            {'' if self.curve_point else '#'}set arrow from {self.curve_point}, graph 0 to {self.curve_point}, graph 1 nohead ls 7


            
            #discover all per-LM .dat files
            files  = "{files_str}"
            N      = words(files)
            labels = "{labels}"
            
            # place at a fixed screen‐coordinate outside the top‐right of the plot
            #set label 1 sprintf("log(WER)=%.2f+%.2f log(PPL)", b,a) \
            #at screen 0.95,0.03 right
            
            #finally, the plot command #sprintf("Regression: log(W)=%.2f+%.2f log(P)", a,b),
            plot \
              for [i=1:N] word(files,i) using 1:2 \
                  with points ls i title sprintf("%s", word(labels,i)), \
              {f'[5:{self.curve_point}] ' if self.curve_point else ''}f_real(x) with lines ls 6 title "Regression-1",\
              {f'[{self.curve_point}:185] f1_real(x) with lines ls 6 title "Regression-2"' if self.curve_point else ''}

        """)

        with open(plt_path, 'w') as f:
            f.write(gp)

        sp.run(["gnuplot", plt_path], check=True)

    def run(self):
        self.split_dat()
        for dataset_key in self.eval_dataset_keys:
            print(f"\n\n\tPloting: {dataset_key}")
            self.plot_gnuplot(dataset_key)
