"""Training-curve plots over ``metrics.train.jsonl`` (backlog D1, F3).

The point of this job is the OVERLAY. Training loss and the in-loop knowledge probe are logged to
the same file but tell opposite stories: loss descends smoothly while factual recall craters in the
first few hundred steps (see ``finetuning.md`` -- collapse is front-loaded and training loss is blind
to it). Reading them apart is how "the loss looks fine" survived as long as it did, so the top panel
puts them on one step axis with twin y-axes.

Below that sit the schedule and per-module panels (backlog F3): the learning rate the optimizer
actually applied, the pre-clip gradient norm against its clip threshold, and the per-module gradient
and weight-displacement curves that localise a collapse to one of the four stacks.

A permanent Job rather than a throwaway script, per the standing rule: it parses durable outputs, so
it regenerates for free on new runs and costs nothing to re-point at a different set of arms. It is
a login-node ``mini_task`` reading finished files -- it never re-runs the training it summarises.

Row shapes in ``metrics.train.jsonl`` (all handled; a run predating any of them simply renders fewer
panels rather than failing, so the job is safe to point at any arm):
    {"step": 1, "loss": 2.34, "lr": 2e-9, "lr_scale": 0.002, "grad_norm": 7.75, "grad_clip": 1.0,
     "grad_clipped": true, "grad_rms_by_module": {...}, "module_numel": {...}}
    {"kind": "knowledge", "step": 100, "knowledge_accuracy": 0.156, "knowledge_avg_quality": 1.31}
    {"kind": "knowledge_error", "step": 900, "error": "OSError(5, 'Input/output error')"}

⚠ **Per-module norms are plotted ONLY in their normalised, per-weight form.** A raw
``grad_norm_by_module`` is an L2 sum over a bucket, so it scales with sqrt(parameter count): on a
real LoRA checkpoint the depformer holds 23x the text head's weights and reads as moving ~5x harder
at identical per-weight motion. Plotting the raw numbers side by side is precisely the misreading
that produced a wrong finding on 2026-08-05. If a file carries raw norms but no ``module_numel``
(runs started before that field existed), the panel is SKIPPED with the reason drawn on the figure --
never silently filled with raw values. Pass ``module_numel={label: {bucket: count}}`` to plot those
runs; the counts come from the LoRA checkpoint's safetensors header.

⚠ The file is APPEND-ordered, not STEP-ordered. A preempted run resumes from its last complete
checkpoint and replays every step between that checkpoint and where it died, so the step counter
runs backwards mid-file (observed on ``a8_fast``: step 1930 -> 1510, 43 steps replayed). Plotting
the raw append order draws a line travelling right-to-left across the figure -- which is what a
reader notices, and which silently double-counts the replayed steps in the summary stats. See
``_split_at_resumes`` / ``_canonical``.
"""

import json
import math

from sisyphus import Job, Task, tk

#: Rows without this key are ordinary training rows (loss/lr); probe rows set it to "knowledge".
KIND_KEY = "kind"
KNOWLEDGE_KIND = "knowledge"
#: A probe step that failed every retry. Written so a hole in the trajectory is present IN the data
#: rather than merely absent from it -- absence is what let a8_long look measured to step 6000.
KNOWLEDGE_ERROR_KIND = "knowledge_error"

#: Module buckets in the order they are drawn, matching ``train_loop.MODULE_BUCKETS``. ``other`` is
#: last and means the parameter naming drifted -- it is plotted so the drift is visible.
MODULE_ORDER = ("text_head", "audio_heads", "depformer", "temporal", "other")
#: One linestyle per bucket, so colour can stay reserved for the arm.
MODULE_STYLE = {
    "text_head": "-",
    "audio_heads": "--",
    "depformer": ":",
    "temporal": "-.",
    "other": (0, (3, 1, 1, 1, 1, 1)),
}


class TrainMetricsPlot(Job):
    """Overlay training loss, the in-loop knowledge probe, and the schedule/per-module metrics.

    ``metrics`` is ``{label -> metrics.train.jsonl Path}``; insertion order fixes both the legend
    order and the job hash, so re-ordering the dict is a different job (deliberate -- the figure
    changes).

    ``origin`` is ``{label -> "ours"|"hf"}``, surfaced in the legend, so a reader can never mistake
    a released checkpoint for something we trained (the standing provenance rule; see
    ``FDB_MODEL_ORIGIN``).

    ``module_numel`` is ``{label -> {bucket -> parameter count}}``, needed only for runs whose
    metrics carry raw per-module norms but not ``module_numel`` (i.e. started between the A2 work
    and the F12 normalisation, both 2026-08-05). Runs from either side of that window need nothing.
    """

    def __init__(
        self,
        *,
        metrics: dict,
        origin: dict | None = None,
        module_numel: dict | None = None,
        title: str = "training curves",
    ):
        self.metrics = metrics
        self.origin = origin or {}
        self.module_numel = module_numel or {}
        self.title = title
        self.out_png = self.output_path("train_curves.png")
        self.out_stats = self.output_path("train_stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _split_at_resumes(steps: list, vals: list) -> list:
        """Split one series wherever the step counter fails to advance.

        Each returned ``(steps, vals)`` run is monotonically increasing, so it can be drawn as a
        polyline without the connecting segment that would otherwise shoot backwards. Every break
        is a resume: the run died and restarted from an earlier checkpoint.
        """
        segments, cur_s, cur_v = [], [], []
        for s, v in zip(steps, vals):
            if cur_s and s <= cur_s[-1]:
                segments.append((cur_s, cur_v))
                cur_s, cur_v = [], []
            cur_s.append(s)
            cur_v.append(v)
        if cur_s:
            segments.append((cur_s, cur_v))
        return segments

    @staticmethod
    def _canonical(steps: list, vals: list) -> tuple:
        """Collapse a replayed series to one value per step, LAST occurrence winning.

        After a resume the same step is logged twice: once by the attempt that died, once by the
        attempt that actually carried the run forward. The later append is the surviving history --
        the earlier one belongs to weights that were rolled back -- so it is the one that must feed
        the curve and the summary stats. Taking the first (or averaging) would report numbers from
        a discarded branch of training.
        """
        by_step = {}
        for s, v in zip(steps, vals):
            by_step[s] = v
        ordered = sorted(by_step)
        return ordered, [by_step[s] for s in ordered]

    @classmethod
    def _per_weight_modules(cls, raw_rows: list, rms_rows: list, numel: dict | None) -> tuple:
        """Resolve the per-module gradient series to its PER-WEIGHT form.

        Returns ``({bucket: (steps, vals)}, note)``. Three cases, in precedence order:

          1. the run logged ``grad_rms_by_module`` already -- use it verbatim;
          2. it logged raw norms plus a usable ``module_numel`` -- divide by ``sqrt(numel)``;
          3. neither -- return no series and a human-readable reason, so the caller draws the reason
             instead of the panel.

        Case 3 must never fall back to the raw values. Raw bucket norms differ by sqrt(parameter
        count), so plotting them together invites exactly the cross-stack comparison that produced
        a wrong finding on 2026-08-05. An empty panel with a stated reason is the honest output.
        """
        if rms_rows:
            buckets = {}
            for b in MODULE_ORDER:
                pts = [(s, d[b]) for s, d in rms_rows if d.get(b) is not None]
                if pts:
                    buckets[b] = cls._canonical([s for s, _ in pts], [v for _, v in pts])
            return buckets, None
        if not raw_rows:
            return {}, "run predates per-module metrics"
        if not numel:
            return {}, "raw per-module norms only -- pass module_numel= to normalise them"
        buckets = {}
        for b in MODULE_ORDER:
            n = numel.get(b)
            if not n:
                continue
            pts = [(s, d[b] / math.sqrt(n)) for s, d in raw_rows if d.get(b) is not None]
            if pts:
                buckets[b] = cls._canonical([s for s, _ in pts], [v for _, v in pts])
        return buckets, None

    @classmethod
    def _relative_deltas(cls, rel_rows: list) -> tuple:
        """``||dtheta|| / ||theta_0||`` per bucket. Already dimensionless, so no numel is needed.

        A ``None`` value is a bucket whose ``theta_0`` had no scale to divide by (LoRA initialises
        its B matrices to zero), and is dropped rather than plotted as a break in the line.
        """
        if not rel_rows:
            return {}, "run predates the relative weight delta"
        buckets = {}
        for b in MODULE_ORDER:
            pts = [(s, d[b]) for s, d in rel_rows if d.get(b) is not None]
            if pts:
                buckets[b] = cls._canonical([s for s, _ in pts], [v for _, v in pts])
        return buckets, None

    @classmethod
    def _parse(cls, path: str, numel_override: dict | None = None) -> dict:
        """Split one metrics file into its series: loss, probe, schedule and per-module.

        Tolerant on purpose: a run that died mid-write leaves a truncated final line, and a partial
        curve is more useful than a crashed plot job. Malformed lines are counted, not raised, and
        the count lands in the stats json so a silently-truncated file is still visible.
        """
        loss_steps, loss_vals = [], []
        acc_steps, acc_vals, qual_vals = [], [], []
        lr_steps, lr_vals = [], []
        gn_steps, gn_vals, clipped_steps = [], [], []
        grad_clip = None
        raw_mod, rms_mod, rel_delta = [], [], []
        file_numel = None
        probe_errors = []
        probe_seconds = []
        bad = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                step = row.get("step")
                if step is None:
                    bad += 1
                    continue
                if row.get(KIND_KEY) == KNOWLEDGE_ERROR_KIND:
                    probe_errors.append({"step": step, "error": row.get("error")})
                elif row.get(KIND_KEY) == KNOWLEDGE_KIND:
                    if row.get("knowledge_accuracy") is not None:
                        acc_steps.append(step)
                        acc_vals.append(float(row["knowledge_accuracy"]) * 100.0)
                        qual_vals.append(float(row.get("knowledge_avg_quality", float("nan"))))
                    if row.get("probe_seconds") is not None:
                        probe_seconds.append(float(row["probe_seconds"]))
                elif row.get("loss") is not None:
                    loss_steps.append(step)
                    loss_vals.append(float(row["loss"]))
                    # lr is the rate actually applied. lr_scale alone is identical across arms with
                    # different base LRs, which is exactly why it was never enough (see train_loop).
                    if row.get("lr") is not None:
                        lr_steps.append(step)
                        lr_vals.append(float(row["lr"]))
                    if row.get("grad_norm") is not None:
                        gn_steps.append(step)
                        gn_vals.append(float(row["grad_norm"]))
                        if row.get("grad_clipped"):
                            clipped_steps.append(step)
                    if row.get("grad_clip") is not None:
                        grad_clip = float(row["grad_clip"])
                    if row.get("module_numel"):
                        file_numel = row["module_numel"]
                    if row.get("grad_rms_by_module"):
                        rms_mod.append((step, row["grad_rms_by_module"]))
                    if row.get("grad_norm_by_module"):
                        raw_mod.append((step, row["grad_norm_by_module"]))
                    if row.get("weight_delta_rel_by_module"):
                        rel_delta.append((step, row["weight_delta_rel_by_module"]))

        # A resume is visible in the loss series (logged every few steps) long before the probe
        # series (every ~100), so the loss stream is the one to detect it on.
        resumes = [
            {"died_at_step": loss_steps[i - 1], "resumed_from_step": s}
            for i, s in enumerate(loss_steps)
            if i and s <= loss_steps[i - 1]
        ]
        c_loss_steps, c_loss = cls._canonical(loss_steps, loss_vals)
        c_acc_steps, c_acc = cls._canonical(acc_steps, acc_vals)
        c_lr_steps, c_lr = cls._canonical(lr_steps, lr_vals)
        c_gn_steps, c_gn = cls._canonical(gn_steps, gn_vals)

        numel = file_numel or numel_override
        grad_modules, grad_note = cls._per_weight_modules(raw_mod, rms_mod, numel)
        delta_modules, delta_note = cls._relative_deltas(rel_delta)

        return {
            "loss_segments": cls._split_at_resumes(loss_steps, loss_vals),
            "probe_segments": cls._split_at_resumes(acc_steps, acc_vals),
            "loss_steps": c_loss_steps,
            "loss": c_loss,
            "probe_steps": c_acc_steps,
            "probe_accuracy": c_acc,
            "probe_quality": qual_vals,
            "probe_seconds": probe_seconds,
            "lr_steps": c_lr_steps,
            "lr": c_lr,
            "grad_norm_steps": c_gn_steps,
            "grad_norm": c_gn,
            "grad_clip": grad_clip,
            "clipped_steps": sorted(set(clipped_steps)),
            "grad_modules": grad_modules,
            "grad_modules_note": grad_note,
            "delta_modules": delta_modules,
            "delta_modules_note": delta_note,
            "module_numel": numel,
            "resumes": resumes,
            "probe_errors": probe_errors,
            # How many logged points were superseded by a replay -- i.e. how much of the file
            # describes weights that were rolled back.
            "superseded_loss_points": len(loss_steps) - len(c_loss_steps),
            "superseded_probe_points": len(acc_steps) - len(c_acc_steps),
            "malformed_lines": bad,
        }

    # ------------------------------------------------------------------------------------ panels
    def _panel_loss_probe(self, ax, series, cmap):
        ax_acc = ax.twinx()
        for i, (label, s) in enumerate(series.items()):
            colour = cmap(i % 10)
            ours = self.origin.get(label, "ours") == "ours"
            tag = f"{'● ' if ours else '○ '}{label}"
            # The canonical (post-resume) history is the curve; the replayed-over attempt is drawn
            # faintly behind it so a resume is visible rather than silently dropped.
            for seg_steps, seg_vals in s["loss_segments"][:-1]:
                ax.plot(seg_steps, seg_vals, color=colour, lw=0.8, alpha=0.22)
            if s["loss_steps"]:
                ax.plot(s["loss_steps"], s["loss"], color=colour, lw=1.2, alpha=0.75, label=f"{tag} loss")
            for seg_steps, seg_vals in s["probe_segments"][:-1]:
                ax_acc.plot(seg_steps, seg_vals, color=colour, lw=1.0, alpha=0.22, linestyle=":")
            if s["probe_steps"]:
                ax_acc.plot(
                    s["probe_steps"],
                    s["probe_accuracy"],
                    color=colour,
                    lw=2.0,
                    marker="o" if ours else "s",
                    markerfacecolor=colour if ours else "none",
                    linestyle="--",
                    label=f"{tag} knowledge %",
                )
            # A failed probe step is marked where it happened, so a gap in the dashed line reads as
            # "the probe died here" rather than "recall went unmeasured for no stated reason".
            for e in s["probe_errors"]:
                ax_acc.axvline(e["step"], color=colour, lw=1.4, alpha=0.6, linestyle=":")
            for r in s["resumes"]:
                ax.axvline(r["resumed_from_step"], color=colour, lw=0.9, alpha=0.45, linestyle="-.")
        ax.set_ylabel("training loss")
        ax_acc.set_ylabel("knowledge probe (%)")
        ax_acc.set_ylim(bottom=0)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_acc.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", framealpha=0.9)

    def _panel_lr(self, ax, series, cmap):
        for i, (label, s) in enumerate(series.items()):
            colour = cmap(i % 10)
            if s["lr_steps"]:
                ax.plot(s["lr_steps"], s["lr"], color=colour, lw=1.4, label=label)
            # Where clipping was active the update direction is kept but the magnitude renormalised.
            # Shading it against the LR answers "did the schedule change here, or the gradient?".
            for st in s["clipped_steps"]:
                ax.axvline(st, color=colour, lw=0.6, alpha=0.10)
        ax.set_yscale("log")
        ax.set_ylabel("learning rate")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    def _panel_grad_norm(self, ax, series, cmap):
        for i, (label, s) in enumerate(series.items()):
            colour = cmap(i % 10)
            if s["grad_norm_steps"]:
                ax.plot(s["grad_norm_steps"], s["grad_norm"], color=colour, lw=1.0, alpha=0.8, label=label)
            if s["grad_clip"]:
                ax.axhline(s["grad_clip"], color=colour, lw=0.9, alpha=0.5, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylabel("grad norm (pre-clip)")
        ax.legend(
            fontsize=7,
            loc="upper right",
            framealpha=0.9,
            title="-- = grad_clip",
            title_fontsize=7,
        )

    def _panel_modules(self, ax, series, cmap, key, note_key, ylabel):
        drew = False
        for i, (label, s) in enumerate(series.items()):
            colour = cmap(i % 10)
            for bucket, (steps, vals) in s[key].items():
                drew = True
                ax.plot(
                    steps,
                    vals,
                    color=colour,
                    lw=1.3,
                    linestyle=MODULE_STYLE.get(bucket, "-"),
                    label=f"{label} · {bucket}" if len(series) > 1 else bucket,
                )
        ax.set_ylabel(ylabel)
        if drew:
            ax.set_yscale("log")
            ax.legend(fontsize=6, loc="upper right", ncol=max(1, len(series)), framealpha=0.9)
        else:
            # Say WHY it is empty. "No per-module data" is a fact a reader can act on; a blank
            # panel is one they will assume is a bug in the plot.
            notes = {s[note_key] for s in series.values() if s[note_key]}
            ax.text(
                0.5,
                0.5,
                "no data: " + "; ".join(sorted(notes)),
                ha="center",
                va="center",
                fontsize=8,
                color="0.4",
                transform=ax.transAxes,
            )
            ax.set_yticks([])

    def run(self):
        import matplotlib

        matplotlib.use("Agg")  # headless login node: no display, must be set before pyplot
        import matplotlib.pyplot as plt

        series = {label: self._parse(p.get(), self.module_numel.get(label)) for label, p in self.metrics.items()}
        cmap = plt.get_cmap("tab10")

        # Only build panels some arm can fill. A panel no arm has data for is dropped rather than
        # rendered empty -- except the per-module ones, which draw the REASON they are empty,
        # because "why is this blank" is the question a reader will actually have.
        panels = [("loss + knowledge probe", self._panel_loss_probe)]
        if any(s["lr_steps"] for s in series.values()):
            panels.append(("learning rate (shaded where gradients were clipped)", self._panel_lr))
        if any(s["grad_norm_steps"] for s in series.values()):
            panels.append(("gradient norm", self._panel_grad_norm))
        has_modules = any(s["grad_modules"] or s["delta_modules"] or s["grad_norm_steps"] for s in series.values())
        if has_modules:
            panels.append(
                (
                    "per-module gradient, PER WEIGHT",
                    lambda ax, se, cm: self._panel_modules(
                        ax, se, cm, "grad_modules", "grad_modules_note", "grad RMS / weight"
                    ),
                )
            )
            panels.append(
                (
                    "per-module weight displacement, RELATIVE",
                    lambda ax, se, cm: self._panel_modules(
                        ax, se, cm, "delta_modules", "delta_modules_note", "||Δθ|| / ||θ₀||"
                    ),
                )
            )

        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.1 * len(panels)), sharex=True, squeeze=False)
        axes = [a[0] for a in axes]
        for ax, (name, draw) in zip(axes, panels):
            draw(ax, series, cmap)
            ax.grid(alpha=0.25)
            ax.set_title(name, fontsize=9, loc="left")
        axes[-1].set_xlabel("training step")

        n_resumes = sum(len(s["resumes"]) for s in series.values())
        subtitle = "top: solid = loss (left), dashed = knowledge probe (right)"
        if n_resumes:
            subtitle += f"; {n_resumes} resume(s) marked -.- , replayed steps faded"
        if has_modules:
            subtitle += "\nper-module curves are NORMALISED per weight; raw L2 norms are NOT comparable across stacks"
        fig.suptitle(f"{self.title}\n{subtitle}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(self.out_png.get(), dpi=150)
        plt.close(fig)

        stats = {}
        for label, s in series.items():
            acc = s["probe_accuracy"]
            stats[label] = {
                "origin": self.origin.get(label, "ours"),
                "n_loss_points": len(s["loss"]),
                "n_probe_points": len(acc),
                "first_loss": s["loss"][0] if s["loss"] else None,
                "final_loss": s["loss"][-1] if s["loss"] else None,
                # peak_probe_step is the actionable number: if recall peaks early and decays, the
                # useful checkpoint is not the last one (backlog A10, early stopping).
                "peak_probe_accuracy": max(acc) if acc else None,
                "peak_probe_step": s["probe_steps"][acc.index(max(acc))] if acc else None,
                "final_probe_accuracy": acc[-1] if acc else None,
                # A preemption/resume record. last_probe_step well below the final loss step means
                # the probe stopped reporting while training continued -- the failure mode that made
                # a8_long look measured to step 6000 when it stopped at 800.
                "last_loss_step": s["loss_steps"][-1] if s["loss_steps"] else None,
                "last_probe_step": s["probe_steps"][-1] if s["probe_steps"] else None,
                "peak_lr": max(s["lr"]) if s["lr"] else None,
                "max_grad_norm": max(s["grad_norm"]) if s["grad_norm"] else None,
                "grad_clip": s["grad_clip"],
                # What fraction of logged steps hit the clip. Harmless under Adam at weight_decay=0
                # (the rescale cancels in m/sqrt(v)) but worth seeing before reading a schedule.
                "clipped_fraction": (
                    len(s["clipped_steps"]) / len(s["grad_norm_steps"]) if s["grad_norm_steps"] else None
                ),
                # The number D3 has been missing: what a probe step actually costs.
                "median_probe_seconds": (
                    sorted(s["probe_seconds"])[len(s["probe_seconds"]) // 2] if s["probe_seconds"] else None
                ),
                "module_numel": s["module_numel"],
                "module_note": s["grad_modules_note"],
                "resumes": s["resumes"],
                "probe_errors": s["probe_errors"],
                "superseded_loss_points": s["superseded_loss_points"],
                "superseded_probe_points": s["superseded_probe_points"],
                "malformed_lines": s["malformed_lines"],
            }
        with open(self.out_stats.get(), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2), flush=True)


def train_metrics_plot_py(
    name: str,
    metrics: dict,
    *,
    origin: dict | None = None,
    module_numel: dict | None = None,
    title: str | None = None,
):
    """Register a training-curve plot under ``output/train_curves/<name>/``.

    ``metrics`` maps a label to a ``SpeechFinetune.out_rundir``; the ``metrics.train.jsonl`` inside
    is resolved here so callers pass the handle they already have rather than knowing the filename.

    ``module_numel`` is only needed for a run logged between the A2 per-module work and the F12
    normalisation (both 2026-08-05), which has raw norms but no parameter counts to divide by.
    """
    resolved = {label: rundir.join_right("metrics.train.jsonl") for label, rundir in metrics.items()}
    job = TrainMetricsPlot(metrics=resolved, origin=origin, module_numel=module_numel, title=title or name)
    tk.register_output(f"train_curves/{name}/plot.png", job.out_png)
    tk.register_output(f"train_curves/{name}/stats.json", job.out_stats)
    return job
