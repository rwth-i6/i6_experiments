import copy
import glob
import numpy as np
import pickle
import os
import os.path as path
import subprocess as sp
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, Literal
import pandas as pd
import re

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
        results: List[Tuple[Dict[str, Union[tk.Variable, float]], tk.Path]],
        # PPL: {dataset_name: ppl}
        # WER: {"best_scores": {dataset name: wer} }

        lm_tunes: List[Optional[tk.Path]],
        prior_tunes: List[Optional[tk.Path]],
        search_errors: List[Dict[str,Optional[tk.Path]]],
        search_errors_rescore: List[Dict[str,Optional[tk.Path]]],
        lm_default_scales: List[Optional[float]] = None,
        prior_default_scales: List[Optional[float]] = None,
        eval_dataset_keys: List[str] = ["test-other","dev-other"],
        include_search_error: bool = True,
        # Reserved for plot setting
    ):
        self.out_summary = self.output_path("summary.csv")
        self.out_plot_folder = self.output_path("plots", directory=True)
        self.out_tabel_folder = self.output_path("tables", directory=True)
        # self.out_plot1 = self.output_path("plots/dev_other.png")
        # self.out_plot2 = self.output_path("plots/test_other.png")
        # self.out_plot3 = self.output_path("plots/dev_clean.png")
        # self.out_plot4 = self.output_path("plots/test_clean.png")
        self.out_plots = [self.output_path(f"plots/{key}.png") for key in eval_dataset_keys]
        self.out_plot_avg = self.output_path(f"plots/avg.png")
        self.out_tables = {key: self.output_path(f"tables/{key}.csv") for key in eval_dataset_keys + ["wers", "ppls", "avg"]}
        self.names = names
        self.results = results

        self.lm_tunes = lm_tunes
        self.prior_tunes = prior_tunes
        self.lm_default_scales = lm_default_scales
        self.prior_default_scales = prior_default_scales
        self.search_errors = search_errors
        self.search_errors_rescore = search_errors_rescore
        self.eval_dataset_keys = eval_dataset_keys
        self.include_search_error = include_search_error

    def tasks(self) -> Iterator[Task]:
        yield Task("create_table", mini_task=True)#, rqmt={"cpu": 1, "time": 1, "mem": 4})
        yield Task("export_dataset_tables", mini_task=True)
        yield Task("export_metric_matrix", mini_task=True)
        yield Task("export_metric_averages", mini_task=True)
        yield Task("plots", mini_task=True)


    @staticmethod
    def find_relevant_key(dict, key):
        for k in dict.keys():
            if key in k or k in key:
                return k
            try:
                minimal_key = key.split(".")[3]
            except IndexError:
                minimal_key = key # Fallback
            if minimal_key in k:
                return k
        raise ValueError(f"Key {key} not found in {dict}")

    def get_points(self):
        for i, (ppl, wers) in enumerate(self.results):
            print(f"{self.names[i]}: ({ppl}, {wers})\n")
        ppls = list()
        wers = list()
        for i, (ppl_dict, wer_path) in enumerate(self.results):
            ppl_dict_ = dict()
            for k, ppl_log in ppl_dict.items():
                """extracts ppl score from the ppl.log file"""
                if all(k not in key_name for key_name in self.eval_dataset_keys):
                    print(f"{k} not in {self.eval_dataset_keys}, skip this PPL report")
                    continue
                if isinstance(ppl_log, float):
                    ppl_dict_[k] = ppl_log
                    print(f"Float ppl -> Got ppl for {i}th entry on {k}: {self.names[i]}")
                elif isinstance(ppl_log, tk.Path):
                    ppl_name_key = "ppl1=" if "gram" in self.names[i] else "ppl="
                    with open(ppl_log.get_path(), "rt") as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split(" ")
                            for idx, ln in enumerate(line):
                                if ln == ppl_name_key or ln == "Perplexity:":
                                    ppl_dict_[k] = float(line[idx + 1])
                        print(f"Log ppl -> Got ppl for {self.names[i]}")
                else:
                    assert isinstance(ppl_log,  tk.Variable), "PPL must be tk.Path or tk.Variable, or raw float"
                    ppl_dict_[k] = float(ppl_log.get())
                    print(f"Tk.var ppl -> Got ppl for {self.names[i]}")
            ppls.append(copy.deepcopy(ppl_dict_))
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))

        assert len(ppls) == len(wers)
        #wers = {"dev-other":[all_res["best_scores"]["dev-other"] for all_res in wers], "test-other":[all_res["best_scores"]["test-other"] for all_res in wers]}
        wers_dict = {key: [all_res["best_scores"][key] for all_res in wers] for key in self.eval_dataset_keys}

        ppls = [{self.find_relevant_key(wers[0]['best_scores'], k): v for k,v in d.items()} for d in ppls]
        ppls_dict = {key: [all_res[self.find_relevant_key(all_res, key)] for all_res in ppls] for key in self.eval_dataset_keys}
        return ppls, wers, ppls_dict, wers_dict

    def plots(self):
        import matplotlib.pyplot as plt
        import scipy.optimize
        # Apply logarithmic transformation to PPL
        *_, ppls, wers= self.get_points()
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

        dataset_count = 0
        ppl_avg = np.array([0.0 for _ in range(len(self.names))])
        wer_avg = np.array([0.0 for _ in range(len(self.names))])
        for i, key in enumerate(self.eval_dataset_keys):
            plot(ppls[key], wers[key], self.out_plots[i].get_path(),self.names)
            dataset_count += 1
            ppl_avg += np.array(ppls[key])
            wer_avg += np.array(wers[key])
        plot(list(ppl_avg/dataset_count), list(wer_avg/dataset_count), self.out_plot_avg.get_path(),self.names)

        # plot(ppls, wers["dev-other"], self.out_plot1.get_path(),self.names)
        # plot(ppls, wers["test-other"], self.out_plot2.get_path(),self.names)
        # plot(ppls, wers["dev-clean"], self.out_plot3.get_path(), self.names)
        # plot(ppls, wers["test-clean"], self.out_plot4.get_path(), self.names)

    @staticmethod
    def _parse_dataset_map(columns: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Map dataset_id -> {"WER": colname, "PPL": colname}
        dataset_id = token immediately before '.ref' in the original column name.
        """
        by_dataset: Dict[str, Dict[str, str]] = {}
        for col in columns:
            m = re.match(r"^(.*)\s+(WER|PPL)$", col)
            if not m:
                continue
            base, metric = m.group(1), m.group(2)
            parts = base.split(".")
            try:
                ref_idx = parts.index("ref")
                dataset_token = parts[ref_idx - 1] if ref_idx > 0 else base
            except ValueError:
                try:
                    ref_idx = parts.index("aptk_leg")
                    dataset_token = parts[ref_idx - 1] if ref_idx > 0 else base
                except ValueError:
                    raise  # do not fallback
            by_dataset.setdefault(dataset_token, {})
            by_dataset[dataset_token][metric] = col
        return by_dataset

    def export_metric_matrix(self):
        """
        Build rotated CSVs:
          - Rows = datasets
          - Columns = model names
          - Values = WER/PPL (min across duplicates if multiple rows per model)
        Also writes a separate rotated table for search_error if requested.
        Returns a dict {metric_name: written_csv_path, ...}
        """
        import os
        import math
        import pandas as pd

        # Helper to parse/clean a metric series
        def _metric_series(df, colname, metric):
            s = df[colname]
            if metric == "WER":
                # strip trailing '%' and coerce to float
                s = s.astype(str).str.rstrip("%").replace({"": None}).astype(float)
            else:
                s = pd.to_numeric(s, errors="coerce")
            return s

        # Collect outputs
        written = {}

        # Load once (both metrics come from same summary)
        df = pd.read_csv(self.out_summary.get_path())
        by_dataset = self._parse_dataset_map(df.columns.tolist())

        # Ensure we have a "Model Name" column to pivot on
        if "Model Name" not in df.columns:
            raise AssertionError("Expected a 'Model Name' column to identify LMs/models.")

        # Build rotated tables for WER and PPL
        for metric in ["WER", "PPL"]:
            # Assemble a tall table with columns: [dataset, model, value]
            tall_rows = []
            for ds in sorted(by_dataset.keys()):
                colname = by_dataset[ds].get(metric)
                if not colname:
                    continue
                s = _metric_series(df, colname, metric)
                # For each row in df, collect (ds, model_name, metric_value)
                sub = pd.DataFrame({
                    "dataset": ds,
                    "model": df["Model Name"],
                    "value": s
                })
                tall_rows.append(sub)

            if not tall_rows:
                # Nothing for this metric; skip
                continue

            tall = pd.concat(tall_rows, ignore_index=True)

            # Drop NaNs so aggregation works cleanly
            tall = tall.dropna(subset=["value"])

            if tall.empty:
                # No valid numbers to write
                out_path = self.out_tables[metric.lower() + "s"].get_path()
                # still write an empty file with header
                pd.DataFrame(columns=["dataset"]).to_csv(out_path, index=False)
                written[metric.lower()] = out_path
                continue

            # If there are multiple rows per (dataset, model), take the **min** (lower is better for both WER/PPL)
            agg = (
                tall
                .groupby(["dataset", "model"], as_index=False)["value"]
                .min()
            )

            # Pivot: rows = dataset, columns = model, values = value
            wide = agg.pivot(index="dataset", columns="model", values="value").sort_index()

            # Write
            out_path = self.out_tables[metric.lower() + "s"].get_path()
            # Ensure dataset is a column
            wide = wide.reset_index()
            wide.to_csv(out_path, index=False)
            written[metric.lower()] = out_path

        # Optional: separate rotated Search Error table
        if self.include_search_error:
            # Try to find per-dataset search_error columns first; otherwise fallback to a single "search_error"
            se_candidates = [c for c in df.columns if c.endswith(" search_error")]
            has_per_dataset = len(se_candidates) > 0

            tall_rows = []
            for ds in sorted(by_dataset.keys()):
                se_col = None
                if has_per_dataset:
                    # Heuristic matching borrowed from original code
                    for c in se_candidates:
                        if ((f".{ds}.ref." in c or (len(c.split(".")) > 1 and c.split(".")[-2] == ds)) or
                                (f".{ds}.apptek_leg." in c or (len(c.split(".")) > 1 and c.split(".")[-2] == ds))):
                            se_col = c
                            break
                if not se_col and "search_error" in df.columns:
                    se_col = "search_error"

                if se_col and se_col in df.columns:
                    s = pd.to_numeric(df[se_col], errors="coerce")
                    sub = pd.DataFrame({
                        "dataset": ds,
                        "model": df["Model Name"],
                        "value": s
                    })
                    tall_rows.append(sub)

            if tall_rows:
                tall = pd.concat(tall_rows, ignore_index=True)
                tall = tall.dropna(subset=["value"])
                if not tall.empty:
                    agg = (
                        tall
                        .groupby(["dataset", "model"], as_index=False)["value"]
                        .min()
                    )
                    wide = agg.pivot(index="dataset", columns="model", values="value").sort_index().reset_index()

                    # Write next to the WER table path (or use dedicated table if available)
                    base_path = self.out_tables.get("wers", None)
                    if base_path is not None:
                        base_path = base_path.get_path()
                        se_out_path = os.path.splitext(base_path)[0] + "__search_error.csv"
                    else:
                        # Fallback generic name
                        se_out_path = os.path.join(os.path.dirname(self.out_summary.get_path()), "search_errors.csv")

                    wide.to_csv(se_out_path, index=False)
                    written["search_error"] = se_out_path

    def export_metric_averages(self):
        """
        Compute per-model averages across dev sets and eval sets separately.

        Output columns:
        model spec + avg_WER_dev + avg_PPL_dev + avg_WER_eval + avg_PPL_eval
        [+ optional avg_search_error_dev, avg_search_error_eval].

        Returns: written CSV path.
        """
        import pandas as pd
        id_cols: Tuple[str, str, str] = ("Model Name", "lm_scale", "prior_scale")
        df = pd.read_csv(self.out_summary.get_path())

        by_dataset = self._parse_dataset_map(df.columns.tolist())
        dev_datasets = [ds for ds in by_dataset if "dev" in ds.lower()]
        eval_datasets = [ds for ds in by_dataset if "eval" in ds.lower()]

        def compute_avg(cols, is_wer=False):
            if not cols:
                return pd.Series([float("nan")] * len(df))
            if is_wer:
                mat = df[cols].astype(str).apply(lambda s: s.str.rstrip("%"))
                mat = mat.apply(pd.to_numeric, errors="coerce")
            else:
                mat = df[cols].apply(pd.to_numeric, errors="coerce")
            return mat.mean(axis=1, skipna=True)

        out = df.loc[:, [c for c in id_cols if c in df.columns]].copy()

        # WER averages
        out["avg_WER_dev"] = compute_avg([by_dataset[ds]["WER"] for ds in dev_datasets if "WER" in by_dataset[ds]],
                                         is_wer=True)
        out["avg_WER_eval"] = compute_avg([by_dataset[ds]["WER"] for ds in eval_datasets if "WER" in by_dataset[ds]],
                                          is_wer=True)

        # PPL averages
        out["avg_PPL_dev"] = compute_avg([by_dataset[ds]["PPL"] for ds in dev_datasets if "PPL" in by_dataset[ds]])
        out["avg_PPL_eval"] = compute_avg([by_dataset[ds]["PPL"] for ds in eval_datasets if "PPL" in by_dataset[ds]])

        if self.include_search_error:
            def compute_se_avg(datasets):
                se_cols = [
                    c for c in df.columns
                    if c.endswith(" search_error") and any(f".{ds}.ref." in c for ds in datasets)
                ]
                if se_cols:
                    se_mat = df[se_cols].apply(pd.to_numeric, errors="coerce")
                    return se_mat.mean(axis=1, skipna=True)
                elif "search_error" in df.columns:
                    return pd.to_numeric(df["search_error"], errors="coerce")
                else:
                    return pd.Series([float("nan")] * len(df))

            out["avg_search_error_dev"] = compute_se_avg(dev_datasets)
            out["avg_search_error_eval"] = compute_se_avg(eval_datasets)

        out_path = self.out_tables["avg"].get_path()
        out.to_csv(out_path, index=False)

    def export_dataset_tables(self):
        """
        Create per-dataset CSVs with columns:
        [Model Name, lm_scale, prior_scale, search_error, WER, PPL]

        Assumes input has columns like:
          "...ref.ff_wer WER" and "...ref.ff_wer PPL"
        The minimal dataset name is the token right before ".ref",
        e.g. "test_set.ES_US.f8kHz.dev_callhome-v4.ref.ff_wer WER"
        -> "dev_callhome-v4".
        """
        df = pd.read_csv(self.out_summary.get_path())
        keep_cols: Tuple[str, str, str, str] = ("Model Name", "lm_scale", "prior_scale", "search_error")
        # Find metric columns and group by dataset id
        metric_cols: List[str] = [c for c in df.columns if c.endswith(" WER") or c.endswith(" PPL")]
        by_dataset: Dict[str, Dict[str, str]] = {}

        for col in metric_cols:
            m = re.match(r"^(.*)\s+(WER|PPL)$", col)
            if not m:
                continue
            base, metric = m.group(1), m.group(2)
            parts = base.split(".")
            try:
                ref_idx = parts.index("ref")
                dataset_token = parts[ref_idx - 1] if ref_idx > 0 else base
            except ValueError:
                try:
                    ref_idx = parts.index("aptk_leg")
                    dataset_token = parts[ref_idx - 1] if ref_idx > 0 else base
                except ValueError:
                    raise  # do not fallback
            dataset_id = dataset_token

            by_dataset.setdefault(dataset_id, {})
            by_dataset[dataset_id][metric] = col

        results: Dict[str, str] = {}
        for dataset_id, metric_map in by_dataset.items():
            cols_to_keep = list(keep_cols)
            if "WER" in metric_map:
                cols_to_keep.append(metric_map["WER"])
            if "PPL" in metric_map:
                cols_to_keep.append(metric_map["PPL"])

            # Skip if neither WER nor PPL present
            if len(cols_to_keep) == len(keep_cols):
                continue

            sub = df.loc[:, [c for c in cols_to_keep if c in df.columns]].copy()
            # Shorten metric headers
            rename_map = {}
            if "WER" in metric_map:
                rename_map[metric_map["WER"]] = "WER"
            if "PPL" in metric_map:
                rename_map[metric_map["PPL"]] = "PPL"
            sub.rename(columns=rename_map, inplace=True)

            # Make sorting sensible: numeric WER/PPL
            if "WER" in sub.columns:
                sub["WER"] = sub["WER"].astype(str).str.rstrip("%").replace({"": None}).astype(float)
            if "PPL" in sub.columns:
                sub["PPL"] = pd.to_numeric(sub["PPL"], errors="coerce")
            sort_cols = [c for c in ["WER", "PPL"] if c in sub.columns]
            if sort_cols:
                sub = sub.sort_values(by=sort_cols, ascending=True, kind="mergesort")

            # safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_id)
            # out_path = os.path.join()
            sub.to_csv(self.out_tables[self.find_relevant_key(self.out_tables,dataset_id)].get_path(), index=False)

    def create_table(self):
        """
        Build a CSV summary of models with PPL/WER and (re)score search errors.
        Robust to attributes being None (attributes that are None are simply ignored).
        """

        # -------- Helpers --------
        import os
        import json
        import csv
        from typing import Any

        def as_path_str(maybe_path_obj: Any) -> str | None:
            """Try to obtain a filesystem path from either a string or objects with .get_path()."""
            if maybe_path_obj is None:
                return None
            if isinstance(maybe_path_obj, str):
                return maybe_path_obj
            get_path = getattr(maybe_path_obj, "get_path", None)
            if callable(get_path):
                return get_path()
            # As a last resort, if it looks like a path on disk:
            if isinstance(maybe_path_obj, os.PathLike):
                return os.fspath(maybe_path_obj)
            return None

        def safe_get(seq, idx, default=None):
            """Index a sequence if possible; return default otherwise."""
            try:
                return seq[idx]
            except Exception:
                return default

        def get_search_error(file_like) -> str:
            """Extract a percent number like '12.3%' from the first line; return '-' if not found."""
            import re
            path = as_path_str(file_like)
            if not path or not os.path.exists(path):
                return "-"
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline()
            m = re.search(r"([-+]?\d*\.\d+|\d+)%", first)
            return m.group(0) if m else "-"

        def retrieve_from_file(file_like, default_scale: float | int | None):
            """
            When tune is a path, load JSON and read 'best_tune' then add to default scale.
            """
            assert default_scale is not None, "default_scale must not be None when using a tune file"
            path = as_path_str(file_like)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Tune file does not exist: {file_like}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return default_scale + data["best_tune"]

        def compute_scale(tune, default_scale):
            """
            Compute the effective scale given either:
            - None -> use default_scale
            - tk.Variable -> float(value)
            - path-like -> default_scale + best_tune (from JSON)
            """
            if tune is None:
                return default_scale
            # Avoid importing tkinter at top; compare by duck-typing to reduce hard dependency.
            is_tk_var = (tune.__class__.__name__ == "Variable" and hasattr(tune, "get"))
            if is_tk_var:
                return float(tune.get())
            path = as_path_str(tune)
            if path:
                return retrieve_from_file(tune, default_scale)
            # Fallback: if someone directly passed a number
            if isinstance(tune, (int, float)):
                return float(tune)
            raise TypeError(f"Unsupported tune type: {type(tune).__name__}")

        def fmt_num(x, decimals=1):
            """Format numbers like 1 decimal; gracefully handle missing or non-numeric."""
            if isinstance(x, (int, float)):
                return f"{x:.{decimals}f}"
            return "-"

        # -------- Gather inputs --------
        # ppls comes from get_points(); keep as-is
        ppls, *_ = self.get_points()  # expecting ppls to be a list of dicts keyed by dataset_key

        # Load WER JSONs (aligned with self.results)
        wers = []
        for _, wer_path in self.results:
            path = as_path_str(wer_path)
            with open(path, "r", encoding="utf-8") as f:
                wers.append(json.load(f))  # expects {'best_scores': {...}, ...}

        # Optional attributes; any of these might be None (the attribute itself)
        lm_tunes = getattr(self, "lm_tunes", None)
        search_errors = getattr(self, "search_errors", None)
        search_errors_rescore = getattr(self, "search_errors_rescore", None)
        lm_default_scales = getattr(self, "lm_default_scales", None)
        prior_tunes = getattr(self, "prior_tunes", None)
        prior_default_scales = getattr(self, "prior_default_scales", None)

        names = list(self.names)  # names for each row/model
        csv_filename = self.out_summary.get_path()

        # -------- Header --------
        dataset_header = []
        for dataset_key in self.eval_dataset_keys:
            dataset_header.extend([
                f"{dataset_key} PPL",
                f"{dataset_key} WER",
                f"{dataset_key} Search Error",
                f"{dataset_key} Search Error (rescore)",
            ])
        table_data = [["Model Name", "lm_scale", "prior_scale"] + dataset_header]

        # -------- Rows --------
        for i, name in enumerate(names):
            ppl_dict = safe_get(ppls, i, default={}) or {}
            wer_obj = safe_get(wers, i, default={}) or {}
            best_scores = wer_obj.get("best_scores", {}) if isinstance(wer_obj, dict) else {}

            lm_tune_item = safe_get(lm_tunes, i) if lm_tunes is not None else None
            prior_tune_item = safe_get(prior_tunes, i) if prior_tunes is not None else None

            lm_def = safe_get(lm_default_scales, i) if lm_default_scales is not None else None
            prior_def = safe_get(prior_default_scales, i) if prior_default_scales is not None else None

            # Compute scales (allow lists containing None; only the attribute being None is special-cased)
            lm_scale = compute_scale(lm_tune_item, lm_def)
            prior_scale = compute_scale(prior_tune_item, prior_def)

            # Optional policy: if lm_scale is 0, you may want prior_scale = 0 (as per your comment)
            # Uncomment if you want that behavior:
            # if lm_scale == 0:
            #     prior_scale = 0

            se_dict = safe_get(search_errors, i, default={}) if search_errors is not None else {}
            se_rescore_dict = safe_get(search_errors_rescore, i,
                                       default={}) if search_errors_rescore is not None else {}

            row = [name, fmt_num(lm_scale, 2), fmt_num(prior_scale, 2)]

            for dataset_key in self.eval_dataset_keys:
                # PPL
                row.append(fmt_num(ppl_dict.get(dataset_key, "-"), 1))
                # WER
                row.append(fmt_num(best_scores.get(dataset_key, "-"), 1))
                # Search Error & Rescore
                se_log = se_dict.get(dataset_key) if isinstance(se_dict, dict) else None
                row.append(get_search_error(se_log))

                se_rescore_log = se_rescore_dict.get(dataset_key) if isinstance(se_rescore_dict, dict) else None
                row.append(get_search_error(se_rescore_log))

            table_data.append(row)

        # -------- Write CSV --------
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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

        def get_idx_name(header:List[str], dataset_key: str, measure: str = "wer"):
            for name in header:
                if dataset_key.lower() in name.lower() and measure.lower() in name.lower():
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
            idx_ppl = header.index(get_idx_name(header, data_setkey, "ppl"))

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
