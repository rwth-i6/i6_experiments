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
"""

import json

from sisyphus import Job, Task, tk

#: Rows without this key are ordinary training rows (loss/lr); probe rows set it to "knowledge".
KIND_KEY = "kind"
KNOWLEDGE_KIND = "knowledge"


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
    def _parse(path: str) -> dict:
        """Split one metrics file into the loss series and the knowledge-probe series.

        Tolerant on purpose: a run that died mid-write leaves a truncated final line, and a partial
        curve is more useful than a crashed plot job. Malformed lines are counted, not raised, and
        the count lands in the stats json so a silently-truncated file is still visible.
        """
        loss_steps, loss_vals = [], []
        acc_steps, acc_vals, qual_vals = [], [], []
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
                if row.get(KIND_KEY) == KNOWLEDGE_KIND:
                    if row.get("knowledge_accuracy") is not None:
                        acc_steps.append(step)
                        acc_vals.append(float(row["knowledge_accuracy"]) * 100.0)
                        qual_vals.append(float(row.get("knowledge_avg_quality", float("nan"))))
                elif row.get("loss") is not None:
                    loss_steps.append(step)
                    loss_vals.append(float(row["loss"]))
        return {
            "loss_steps": loss_steps,
            "loss": loss_vals,
            "probe_steps": acc_steps,
            "probe_accuracy": acc_vals,
            "probe_quality": qual_vals,
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
            if s["loss_steps"]:
                ax_loss.plot(s["loss_steps"], s["loss"], color=colour, lw=1.2, alpha=0.75, label=f"{tag} loss")
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

        ax_loss.set_xlabel("training step")
        ax_loss.set_ylabel("training loss")
        ax_acc.set_ylabel("in-loop knowledge probe (% correct)")
        ax_acc.set_ylim(bottom=0)
        ax_loss.set_title(f"{self.title}\nsolid = loss (left), dashed = knowledge probe (right)")
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
