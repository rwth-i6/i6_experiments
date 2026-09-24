"""
:class:`PlotTextRatioGainJob`: relative WER reduction of text injection over the audio-only baseline
vs. the used text : audio ratio (log x), as a Sis graph node (PNG + PDF + the resolved numbers as JSON).

Cells are resolved like in :class:`i6_experiments.users.zeyer.utils.table_data.WriteTableDataJob`:
a literal, or a ``(Path, key)`` tuple into a recog result JSON, or a Sisyphus ``Variable``.
A point's gain is either a literal percentage or a ``(baseline_cell, injection_cell)`` pair.
Filled marker / solid line = the primary evaluation set, hollow marker / dashed line = the secondary set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sisyphus import Job, Task

from i6_experiments.users.zeyer.utils.table_data import WriteTableDataJob


class PlotTextRatioGainJob(Job):
    # v2: legend anchored further below the x-axis label (the module code is not part of the hash)
    __sis_version__ = 3  # legend anchor -0.28, save pad 0.05

    """
    :param ladders: connected series, each ``{"label", "color", "marker", "points": [
        {"ratio", "primary": gain, "secondary": gain}, ...]}``
    :param groups: single-point groups, each ``{"color", "markers": [..], "points": [
        {"label", "ratio", "primary": gain, "secondary": gain or None}, ...]}``;
        a gain is a float (percent) or ``(baseline_cell, injection_cell)``
    :param xlabel, ylabel, note: axis labels and the marker-convention note (top left)
    :param figsize, legend_ncol, fontsize: layout
    """

    def __init__(
        self,
        *,
        ladders: List[Dict[str, Any]],
        groups: List[Dict[str, Any]],
        xlabel: str,
        ylabel: str = "relative WER reduction [%]",
        note: Optional[str] = None,
        figsize: Tuple[float, float] = (6.4, 5.2),
        legend_ncol: int = 3,
        fontsize: float = 7.0,
        markersize: float = 7.0,
        show_secondary: bool = True,
    ):
        super().__init__()
        self.ladders = ladders
        self.groups = groups
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.note = note
        self.figsize = tuple(figsize)
        self.legend_ncol = legend_ncol
        self.fontsize = fontsize
        self.markersize = markersize
        self.show_secondary = show_secondary
        self.out_png = self.output_path("plot.png")
        self.out_pdf = self.output_path("plot.pdf")
        self.out_json = self.output_path("points.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _gain(g) -> Optional[float]:
        if g is None:
            return None
        if isinstance(g, (int, float)):
            return float(g)
        baseline, injection = g
        b = float(WriteTableDataJob._resolve(baseline))
        i = float(WriteTableDataJob._resolve(injection))
        return 100.0 * (b - i) / b

    def run(self):
        import json
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        resolved: Dict[str, Any] = {"ladders": [], "groups": []}
        fig, ax = plt.subplots(figsize=self.figsize)
        for ladder in self.ladders:
            pts = sorted(
                (
                    (float(p["ratio"]), self._gain(p["primary"]), self._gain(p.get("secondary")))
                    for p in ladder["points"]
                ),
                key=lambda t: t[0],
            )
            resolved["ladders"].append({"label": ladder["label"], "points": pts})
            xs = [p[0] for p in pts]
            ax.plot(
                xs,
                [p[1] for p in pts],
                color=ladder["color"],
                marker=ladder["marker"],
                linestyle="-",
                label=ladder["label"],
            )
            if self.show_secondary and all(p[2] is not None for p in pts):
                ax.plot(
                    xs,
                    [p[2] for p in pts],
                    color=ladder["color"],
                    marker=ladder["marker"],
                    linestyle="--",
                    markerfacecolor="none",
                )
        for group in self.groups:
            out_group = []
            for point, marker in zip(group["points"], group["markers"]):
                r = float(point["ratio"])
                g1 = self._gain(point["primary"])
                g2 = self._gain(point.get("secondary"))
                out_group.append({"label": point["label"], "ratio": r, "primary": g1, "secondary": g2})
                ax.plot(
                    [r],
                    [g1],
                    marker=marker,
                    color=group["color"],
                    linestyle="none",
                    markersize=self.markersize,
                    label=point["label"],
                )
                if self.show_secondary and g2 is not None:
                    ax.plot(
                        [r],
                        [g2],
                        marker=marker,
                        color=group["color"],
                        linestyle="none",
                        markersize=self.markersize,
                        markerfacecolor="none",
                    )
            resolved["groups"].append(out_group)
        ax.set_xscale("log")
        ax.set_xlabel(self.xlabel, fontsize=self.fontsize + 1.5)
        ax.set_ylabel(self.ylabel, fontsize=self.fontsize + 1.5)
        ax.tick_params(labelsize=self.fontsize + 1)
        ax.grid(True, which="both", alpha=0.3)
        if self.note:
            ax.text(0.01, 0.99, self.note, transform=ax.transAxes, fontsize=self.fontsize, va="top")
        legend = ax.legend(
            fontsize=self.fontsize,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=self.legend_ncol,
            frameon=False,
            columnspacing=0.6,
            labelspacing=0.25,
            handletextpad=0.4,
        )
        fig.tight_layout()
        # the legend sits below the axes: include it in the saved bounding box, else it is clipped
        save_opts = dict(bbox_inches="tight", bbox_extra_artists=[legend], pad_inches=0.05)
        fig.savefig(self.out_png.get_path(), dpi=150, **save_opts)
        fig.savefig(self.out_pdf.get_path(), **save_opts)
        with open(self.out_json.get_path(), "w") as f:
            json.dump(resolved, f, indent=2)
            f.write("\n")
