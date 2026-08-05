"""Training-curve plots over ``metrics.train.jsonl`` (backlog D1).

The point of this job is the OVERLAY. Training loss and the in-loop knowledge probe are logged to
the same file but tell opposite stories: loss descends smoothly while factual recall craters in the
first few hundred steps (see ``finetuning.md`` -- collapse is front-loaded and training loss is blind
to it). Reading them apart is how "the loss looks fine" survived as long as it did, so the default
figure puts them on one step axis with twin y-axes.

A permanent Job rather than a throwaway script, per the standing rule: it parses durable outputs, so
it regenerates for free on new runs and costs nothing to re-point at a different set of arms. It is
a login-node ``mini_task`` reading finished files -- it never re-runs the training it summarises.

Two row shapes live in ``metrics.train.jsonl`` and both are handled:
    {"step": 1, "loss": 2.34, "lr_scale": 0.002, "percent_done": 0.03, "eta_in_seconds": 425365}
    {"kind": "knowledge", "step": 100, "knowledge_accuracy": 0.156, "knowledge_avg_quality": 1.31,
     "knowledge_n": 64}
Only the probe rows carry ``kind``; a run without the probe simply renders no accuracy series
rather than failing, so the job is safe to point at any arm.

⚠ The file is APPEND-ordered, not STEP-ordered. A preempted run resumes from its last complete
checkpoint and replays every step between that checkpoint and where it died, so the step counter
runs backwards mid-file (observed on ``a8_fast``: step 1930 -> 1510, 43 steps replayed). Plotting
the raw append order draws a line travelling right-to-left across the figure -- which is what a
reader notices, and which silently double-counts the replayed steps in the summary stats. See
``_split_at_resumes`` / ``_canonical``.
"""

import json

from sisyphus import Job, Task, tk

#: Rows without this key are ordinary training rows (loss/lr); probe rows set it to "knowledge".
KIND_KEY = "kind"
KNOWLEDGE_KIND = "knowledge"
#: A probe step that failed every retry. Written so a hole in the trajectory is present IN the data
#: rather than merely absent from it -- absence is what let a8_long look measured to step 6000.
KNOWLEDGE_ERROR_KIND = "knowledge_error"


class TrainMetricsPlot(Job):
    """Overlay training loss and the in-loop knowledge probe for one or more finetune arms.

    ``metrics`` is ``{label -> metrics.train.jsonl Path}``; insertion order fixes both the legend
    order and the job hash, so re-ordering the dict is a different job (deliberate -- the figure
    changes).

    ``origin`` is ``{label -> "ours"|"hf"}``, surfaced in the legend, so a reader can never mistake
    a released checkpoint for something we trained (the standing provenance rule; see
    ``FDB_MODEL_ORIGIN``).
    """

    def __init__(self, *, metrics: dict, origin: dict | None = None, title: str = "training curves"):
        self.metrics = metrics
        self.origin = origin or {}
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
    def _parse(cls, path: str) -> dict:
        """Split one metrics file into the loss series and the knowledge-probe series.

        Tolerant on purpose: a run that died mid-write leaves a truncated final line, and a partial
        curve is more useful than a crashed plot job. Malformed lines are counted, not raised, and
        the count lands in the stats json so a silently-truncated file is still visible.
        """
        loss_steps, loss_vals = [], []
        acc_steps, acc_vals, qual_vals = [], [], []
        probe_errors = []
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
                elif row.get("loss") is not None:
                    loss_steps.append(step)
                    loss_vals.append(float(row["loss"]))

        # A resume is visible in the loss series (logged every few steps) long before the probe
        # series (every ~100), so the loss stream is the one to detect it on.
        resumes = [
            {"died_at_step": loss_steps[i - 1], "resumed_from_step": s}
            for i, s in enumerate(loss_steps)
            if i and s <= loss_steps[i - 1]
        ]
        c_loss_steps, c_loss = cls._canonical(loss_steps, loss_vals)
        c_acc_steps, c_acc = cls._canonical(acc_steps, acc_vals)
        return {
            "loss_segments": cls._split_at_resumes(loss_steps, loss_vals),
            "probe_segments": cls._split_at_resumes(acc_steps, acc_vals),
            "loss_steps": c_loss_steps,
            "loss": c_loss,
            "probe_steps": c_acc_steps,
            "probe_accuracy": c_acc,
            "probe_quality": qual_vals,
            "resumes": resumes,
            "probe_errors": probe_errors,
            # How many logged points were superseded by a replay -- i.e. how much of the file
            # describes weights that were rolled back.
            "superseded_loss_points": len(loss_steps) - len(c_loss_steps),
            "superseded_probe_points": len(acc_steps) - len(c_acc_steps),
            "malformed_lines": bad,
        }

    def run(self):
        import matplotlib

        matplotlib.use("Agg")  # headless login node: no display, must be set before pyplot
        import matplotlib.pyplot as plt

        series = {label: self._parse(p.get()) for label, p in self.metrics.items()}

        fig, ax_loss = plt.subplots(figsize=(11, 6))
        ax_acc = ax_loss.twinx()
        cmap = plt.get_cmap("tab10")

        for i, (label, s) in enumerate(series.items()):
            colour = cmap(i % 10)
            # Provenance in the legend itself: filled marker + bold for ours, hollow for released.
            ours = self.origin.get(label, "ours") == "ours"
            tag = f"{'● ' if ours else '○ '}{label}"

            # The canonical (post-resume) history is the curve; the replayed-over attempt is drawn
            # faintly behind it so a resume is visible rather than silently dropped.
            for seg_steps, seg_vals in s["loss_segments"][:-1]:
                ax_loss.plot(seg_steps, seg_vals, color=colour, lw=0.8, alpha=0.22)
            if s["loss_steps"]:
                ax_loss.plot(s["loss_steps"], s["loss"], color=colour, lw=1.2, alpha=0.75, label=f"{tag} loss")
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
            for r in s["resumes"]:
                ax_loss.axvline(r["resumed_from_step"], color=colour, lw=0.9, alpha=0.45, linestyle="-.")

        ax_loss.set_xlabel("training step")
        ax_loss.set_ylabel("training loss")
        ax_acc.set_ylabel("in-loop knowledge probe (% correct)")
        ax_acc.set_ylim(bottom=0)
        n_resumes = sum(len(s["resumes"]) for s in series.values())
        subtitle = "solid = loss (left), dashed = knowledge probe (right)"
        if n_resumes:
            subtitle += f"; {n_resumes} resume(s) marked -.- , replayed steps faded"
        ax_loss.set_title(f"{self.title}\n{subtitle}")
        ax_loss.grid(alpha=0.25)

        # One merged legend: two axes would otherwise render two boxes that overlap.
        h1, l1 = ax_loss.get_legend_handles_labels()
        h2, l2 = ax_acc.get_legend_handles_labels()
        if h1 or h2:
            ax_loss.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", framealpha=0.9)

        fig.tight_layout()
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
                "resumes": s["resumes"],
                "probe_errors": s["probe_errors"],
                "superseded_loss_points": s["superseded_loss_points"],
                "superseded_probe_points": s["superseded_probe_points"],
                "malformed_lines": s["malformed_lines"],
            }
        with open(self.out_stats.get(), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats, indent=2), flush=True)


def train_metrics_plot_py(name: str, metrics: dict, *, origin: dict | None = None, title: str | None = None):
    """Register a training-curve plot under ``output/train_curves/<name>/``.

    ``metrics`` maps a label to a ``SpeechFinetune.out_rundir``; the ``metrics.train.jsonl`` inside
    is resolved here so callers pass the handle they already have rather than knowing the filename.
    """
    resolved = {label: rundir.join_right("metrics.train.jsonl") for label, rundir in metrics.items()}
    job = TrainMetricsPlot(metrics=resolved, origin=origin, title=title or name)
    tk.register_output(f"train_curves/{name}/plot.png", job.out_png)
    tk.register_output(f"train_curves/{name}/stats.json", job.out_stats)
    return job
