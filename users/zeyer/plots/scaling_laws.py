from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Tuple, Dict
from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


TNumber = Union[tk.Variable, DelayedBase]


class ScalingLawPlotJob(Job):
    """
    Show some scatter points, including optional baselines, and compute & plot the Pareto front.
    Here the left-bottom is better (min x, min y).
    """

    __sis_version__ = 10

    def __init__(
        self,
        *,
        x_label: str,
        y_label: str,
        x_scale: str = "log",
        y_scale: str = "linear",
        baselines: Optional[Dict[str, Union[TNumber, Dict[str, Any]]]] = None,
        points: Dict[str, Union[Sequence[Tuple[TNumber, TNumber]], Dict[str, Any]]],
        filter_outliers: bool = False,
        figsize: Tuple[float, float] = (8, 6),
    ):
        """
        :param x_label: label for x-axis
        :param y_label: label for y-axis
        :param x_scale: scale for x-axis (e.g., 'linear', 'log')
        :param y_scale: scale for y-axis
        :param baselines: name -> y-value
        :param points: name -> list of (x, y) points
        :param filter_outliers: whether to filter out outliers in y-axis
        :param figsize: figure size (width, height)
        """
        super().__init__()

        self.x_label = x_label
        self.y_label = y_label
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.baselines = baselines
        self.points = points
        self.filter_outliers = filter_outliers
        self.figsize = figsize

        self.out_plot_pdf = self.output_path("scaling_laws.pdf")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np

        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)

        name = "Set1"
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
                xy = data_points.pop("xy")
            elif "x" in data_points and "y" in data_points:
                x_data = data_points.pop("x")
                y_data = data_points.pop("y")
                xy = list(zip(x_data, y_data))
            else:
                raise ValueError(f"invalid points {name} : {data_points}")
            if not xy:
                continue

            if self.filter_outliers:
                # Simple outlier removal: remove points that are beyond 1.5 * IQR from Q1 and Q3
                y_data = [y for x, y in xy]
                q1 = np.percentile(y_data, 25)
                q3 = np.percentile(y_data, 75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                xy = [(x, y) for x, y in xy if y <= upper_bound]
                assert xy, "all points were filtered out as outliers? (should not happen)"

            color = data_points.pop("color", colors[idx])

            xy_clamped_min_out = []
            clamp_x_min = None
            if "clamp_x_min" in data_points:
                clamp_x_min = data_points.pop("clamp_x_min")
                xy_clamped_min_out += [(clamp_x_min, y) for x, y in xy if x < clamp_x_min]
                xy = [(x, y) for x, y in xy if x >= clamp_x_min]
                if not xy:
                    continue
            xy_clamped_max_out = []
            clamp_x_max = None
            if "clamp_x_max" in data_points:
                clamp_x_max = data_points.pop("clamp_x_max")
                xy_clamped_max_out += [(clamp_x_max, y) for x, y in xy if x > clamp_x_max]
                xy = [(x, y) for x, y in xy if x <= clamp_x_max]
                if not xy:
                    continue

            if data_points:
                raise ValueError(f"unexpected extra data points options: {data_points}")

            # Plot the data points with markers and a connecting line
            x_data, y_data = zip(*xy)
            ax.scatter(
                x_data,
                y_data,
                marker="o",
                linestyle="-",
                color=color,
                zorder=2,
                label=name,
            )
            # Compute Pareto front (minimize x, minimize y)
            # combine duplicate x values by keeping the min y
            combined = {}
            for x, y in zip(x_data, y_data):
                combined[x] = min(y, combined.get(x, np.inf))

            # sort by x ascending
            sorted_items = sorted(combined.items(), key=lambda kv: kv[0])
            xs_sorted, ys_sorted = zip(*sorted_items)

            pareto_x = []
            pareto_y = []
            best_y = np.inf
            if xy_clamped_min_out:
                best_y = min(y for x, y in xy_clamped_min_out)
            for x, y in zip(xs_sorted, ys_sorted):
                if y < best_y:
                    pareto_x.append(x)
                    pareto_y.append(y)
                    best_y = y
            if xy_clamped_max_out:
                best_y = min(best_y, min(y for x, y in xy_clamped_max_out))

            # Plot Pareto front
            ax.plot(
                ([clamp_x_min] if xy_clamped_min_out else [])
                + pareto_x
                + [clamp_x_max if xy_clamped_max_out else xs_sorted[-1]],
                ([min(y for x, y in xy_clamped_min_out)] if xy_clamped_min_out else []) + pareto_y + [best_y],
                color=color,
                linewidth=2.5,
                marker="s",
                markersize=3,
                linestyle="-",
                zorder=4,
                # label=f"{name} Pareto front",
            )
            ax.scatter(pareto_x, pareto_y, color=color, edgecolor="k", s=60, zorder=5)

            idx += 1

        # Set the xy-axis to a linear, logarithmic, or whatever scale
        ax.set_xscale(self.x_scale)
        ax.set_yscale(self.y_scale)

        # --- Set labels and titles ---

        # Set the x-axis label with 'Parameters' in bold and 'non-embedding' below
        ax.set_xlabel(self.x_label, fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)

        # Add the 'non-embedding' sub-label with specific positioning and style
        # ax.text(0.5, -0.15, "non-embedding", ha="center", va="center", transform=ax.transAxes, fontsize=18, color="gray")
        ax.legend(fontsize=14)

        # --- Customize ticks and grid ---

        # Set the y-axis ticks to match the image
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.xaxis.minorticks_on()

        # show the minor tick labels
        ax.tick_params(axis="x", which="minor", labelsize=10)

        # Display the plot
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off

        plt.savefig(
            self.out_plot_pdf.get_path(),
            metadata={"CreationDate": None},
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
