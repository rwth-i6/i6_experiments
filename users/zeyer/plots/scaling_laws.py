from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Tuple, Dict
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class ScalingLawPlotJob(Job):
    __sis_version__ = 2

    def __init__(
        self,
        *,
        x_label: str,
        y_label: str,
        x_scale: str = "log",
        y_scale: str = "linear",
        baselines: Optional[Dict[str, Union[tk.Variable, Dict[str, Any]]]] = None,
        points: Dict[str, Union[Sequence[Tuple[tk.Variable, tk.Variable]], Dict[str, Any]]],
        filter_outliers: bool = False,
    ):
        """
        :param baselines: name -> y-value
        :param points: name -> list of (x, y) points
        """
        super().__init__()

        self.x_label = x_label
        self.y_label = y_label
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.baselines = baselines
        self.points = points
        self.filter_outliers = filter_outliers

        self.out_plot_pdf = self.output_path("scaling_laws.pdf")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        import numpy as np

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        name = "Accent"
        cmap = mpl.colormaps[name]
        colors = cmap.colors
        idx = 0

        if self.baselines:
            for name, opts in self.baselines.items():
                if not isinstance(opts, dict):
                    opts = {"y": opts}
                opts = instanciate_delayed_copy(opts)
                opts.setdefault("label", name)
                opts.setdefault("color", colors[idx])
                opts.setdefault("linestyle", "--")
                opts.setdefault("zorder", 1)
                ax.axhline(**opts)
                idx += 1

        for name, data_points in self.points.items():
            if not isinstance(data_points, dict):
                data_points = {"xy": data_points}
            data_points = instanciate_delayed_copy(data_points)

            # Unzip the data into separate lists for x and y coordinates
            if "xy" in data_points:
                x_data, y_data = zip(*data_points.pop("xy"))
            elif "x" in data_points and "y" in data_points:
                x_data = data_points.pop("x")
                y_data = data_points.pop("y")
            else:
                raise ValueError(f"invalid points {name} : {data_points}")

            if self.filter_outliers:
                # Simple outlier removal: remove points that are beyond 1.5 * IQR from Q1 and Q3
                q1 = np.percentile(y_data, 25)
                q3 = np.percentile(y_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_points = [(x, y) for x, y in zip(x_data, y_data) if lower_bound <= y <= upper_bound]
                if filtered_points:  # only take this if not empty (should not happen?)
                    x_data, y_data = zip(*filtered_points)

            color = data_points.pop("color", colors[idx])
            if data_points:
                raise ValueError(f"unexpected extra data points options: {data_points}")

            # Plot the data points with markers and a connecting line
            ax.scatter(
                x_data,
                y_data,
                marker="o",
                linestyle="-",
                color=color,
                zorder=2,
                label=name,
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
                color=color,
                linewidth=2.5,
                marker="s",
                markersize=6,
                linestyle="-",
                zorder=4,
                label=f"{name} pareto front",
            )
            ax.scatter(pareto_x, pareto_y, color=color, edgecolor="k", s=60, zorder=5)

            idx += 1

        # Set the xy-axis to a linear, logarithmic, or whatever scale
        ax.set_xscale(self.x_scale)
        ax.set_yscale(self.y_scale)

        # --- Set labels and titles ---

        # Set the x-axis label with 'Parameters' in bold and 'non-embedding' below
        ax.set_xlabel(self.x_label, fontsize=20, fontweight="bold", labelpad=10)
        ax.set_ylabel(self.y_label, fontsize=20, fontweight="bold", labelpad=10)

        # Add the 'non-embedding' sub-label with specific positioning and style
        # ax.text(0.5, -0.15, "non-embedding", ha="center", va="center", transform=ax.transAxes, fontsize=18, color="gray")
        ax.legend(fontsize=14)

        # --- Customize ticks and grid ---

        # Set the y-axis ticks to match the image
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=5))

        # show the minor tick labels
        ax.tick_params(axis="x", which="minor", labelsize=10)

        # Display the plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off

        plt.savefig(
            self.out_plot_pdf.get_path(),
            metadata={"CreationDate": None},
        )
        plt.close()
