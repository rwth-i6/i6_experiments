from sisyphus import Job, Task


class ScalingLawPlotJob(Job):
    def __init__(self):
        super().__init__()

        self.out_plot_pdf = self.output_path("scaling_laws.pdf")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        import numpy as np
        import os

        # no. just glob all scaling law experiments
        import glob

        models = glob.glob("base-scalingLaws-*", root_dir=p_exp)
        print(models)

        metric = "total_train_time_secs"

        def parse_metric(source_path: str) -> int:
            pp = experiment_to_path(source_path)
            pp = os.path.join(pp, f"../{metric}.txt")
            with open(pp) as f:
                return int(f.read().strip())

        ds_name = "test-other"

        def name_to_number(source_path: str) -> float:
            vals = parse_experiment(source_path)
            return vals["DLM sum"][ds_name]

        asrbaseline = parse_experiment(
            "train_data/asrbaseline-(L16-D1024-spm10k-auxAED-b100k-tts)/eval_datasets/input/"
        )[
            "DSR"  # hack for "score_results.txt"
        ][ds_name]
        print(asrbaseline)

        new_models = []
        for name in models:
            try:
                new_models.append((parse_metric(name), name_to_number(name)))
            except Exception as e:
                print(f"Error processing {name}: {e}")
        models = new_models
        # --- User-supplied data ---
        # (x, y)
        data_points = [(x, y) for x, y in models]

        # sort by x
        data_points.sort(key=lambda pair: pair[0])
        print(data_points)

        # Unzip the data into separate lists for x and y coordinates
        x_data, y_data = zip(*data_points)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.axhline(y=asrbaseline, color="red", linestyle="--", label="ASR Baseline", zorder=1)

        # Plot the data points with markers and a connecting line
        ax.scatter(
            x_data,
            y_data,
            marker="o",
            linestyle="-",
            color="#1f77b4",
            zorder=2,
            label="Scaling Law Experiments",
        )
        # Compute Pareto front (minimize x, maximize y)
        # combine duplicate x values by keeping the max y
        combined = {}
        for x, y in zip(x_data, y_data):
            combined[x] = min(y, combined.get(x, np.inf))

        # sort by x ascending
        sorted_items = sorted(combined.items(), key=lambda kv: kv[0])
        xs_sorted, ys_sorted = zip(*sorted_items)

        pareto_x = []
        pareto_y = []
        best_y = np.inf
        for x, y in zip(xs_sorted, ys_sorted):
            if y < best_y:
                pareto_x.append(x)
                pareto_y.append(y)
                best_y = y

        # Plot Pareto front
        ax.plot(
            pareto_x,
            pareto_y,
            color="orange",
            linewidth=2.5,
            marker="s",
            markersize=6,
            linestyle="-",
            zorder=4,
            label="Pareto front",
        )
        ax.scatter(pareto_x, pareto_y, color="orange", edgecolor="k", s=60, zorder=5)

        # Set the x-axis to a logarithmic scale
        ax.set_xscale("log")

        # --- Set labels and titles ---

        # Set the x-axis label with 'Parameters' in bold and 'non-embedding' below
        ax.set_xlabel(
            {"num_params": "Parameters", "total_train_time_secs": "Training time"}.get(metric, metric),
            fontsize=20,
            fontweight="bold",
            labelpad=10,
        )
        # Add the 'non-embedding' sub-label with specific positioning and style
        # ax.text(0.5, -0.15, "non-embedding", ha="center", va="center", transform=ax.transAxes, fontsize=18, color="gray")
        ax.legend(fontsize=14)

        # --- Customize ticks and grid ---

        # Set the y-axis ticks to match the image
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=5))

        # Add this line to show the minor tick labels
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.xaxis.get_minor_formatter().set_useOffset(False)
        ax.xaxis.get_minor_formatter().set_scientific(True)

        ax.tick_params(axis="x", which="minor", labelsize=10)

        # Display the plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(
            self.out_plot_pdf.get_path(),
            metadata={"CreationDate": None},
        )
        plt.close()
