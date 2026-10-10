import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from sisyphus import Job, Task, tk
    from i6_experiments.users.sheremeta.returnn.checkpoint import resolve_label, resolve_names
except ImportError:
    Job, Task, tk = object, None, None
    resolve_label = resolve_names = None

from .throughput import rank_zero_log_text


class _EpochData:
    """Mirror of returnn's EpochData so the learning_rates file can be eval-parsed."""

    def __init__(self, *, learningRate=None, error=None, meta=None, learning_rate=None):
        self.learning_rate = (
            learning_rate if learning_rate is not None else learningRate
        )
        error = error or {}
        if meta is None:
            meta = {
                k[len(":meta:") :]: v
                for k, v in error.items()
                if k.startswith(":meta:")
            }
            error = {k: v for k, v in error.items() if not k.startswith(":meta:")}
        self.error = error
        self.meta = meta


def parse_training_time(learning_rates_text: str) -> Dict[str, Any]:
    """
    Sums the per-epoch compute time of a learning_rates file.

    :param learning_rates_text: the file body
    :return: the training time in seconds, the epochs done, the last step and the device
    """
    import numpy

    data = eval(
        learning_rates_text,
        {
            "EpochData": _EpochData,
            "nan": float("nan"),
            "inf": float("inf"),
            "np": numpy,
            "numpy": numpy,
        },
    )
    total_s = 0.0
    steps = 0
    device = None
    for ep in sorted(data):
        meta = data[ep].meta
        t = meta.get("epoch_train_time_secs")
        if t is not None:
            total_s += float(t)
        steps = max(
            steps,
            int(
                meta.get("global_train_step_end", meta.get("global_train_step", 0)) or 0
            ),
        )
        device = meta.get("device", device)
    return {
        "train_time_s": total_s,
        "epochs_done": len(data),
        "total_steps": steps,
        "device": device,
    }


_UNIT = {
    "": 1.0,
    "K": 1024.0,
    "M": 1024.0**2,
    "G": 1024.0**3,
    "T": 1024.0**4,
    "P": 1024.0**5,
}
_ALLOC_RE = re.compile(r"alloc peak\s+([0-9.]+)\s*([KMGTP]?)i?B", re.IGNORECASE)
_RESERVED_RE = re.compile(r"reserved peak\s+([0-9.]+)\s*([KMGTP]?)i?B", re.IGNORECASE)
_MEMUSAGE_RE = re.compile(r"mem_usage:\S+\s+([0-9.]+)\s*([KMGTP]?)i?B", re.IGNORECASE)
_GRAPH_POOL_RE = re.compile(r"mem_graph_pool:\S+\s+([0-9.]+)\s*([KMGTP]?)i?B", re.IGNORECASE)


def parse_peak_mem_gb(log_text: str) -> Dict[str, Optional[float]]:
    """
    Reads the peak GPU memory of a training from the lines RETURNN logs per step and at start-up.

    ``mem_usage`` is the number to report. RETURNN builds it as the allocated peak plus the CUDA graph
    private pool, which the allocated and reserved counters cannot see, so on a captured row the
    start-up ``alloc peak`` and ``reserved peak`` lines are an order of magnitude too small.

    :param log_text: the first rank's log
    :return: the peaks in GB, None per entry the log does not carry
    """

    def maxof(rx):
        vals = [float(n) * _UNIT[u.upper()] for n, u in rx.findall(log_text)]
        return max(vals) / _UNIT["G"] if vals else None

    return {
        "peak_alloc_gb": maxof(_ALLOC_RE),
        "peak_reserved_gb": maxof(_RESERVED_RE),
        "peak_usage_gb": maxof(_MEMUSAGE_RE),
        "graph_pool_gb": maxof(_GRAPH_POOL_RE),
    }


def count_params(checkpoint_path: str) -> Tuple[int, Dict[str, int]]:
    """
    Counts the parameters of a torch checkpoint, in total and per top-level module.

    :param checkpoint_path: the checkpoint file
    :return: the total and the count per module
    """
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]
    total = 0
    by_comp: Dict[str, int] = {}
    for k, v in state.items():
        if not hasattr(v, "numel"):
            continue
        n = int(v.numel())
        total += n
        comp = k.split(".")[0]
        by_comp[comp] = by_comp.get(comp, 0) + n
    return total, by_comp


def _round(x, n):
    return None if x is None else round(x, n)


def parse_benchmark(benchmark_text: str) -> Dict[str, Any]:
    """
    Reads a benchmark.json into the RTF, memory and throughput columns.

    :param benchmark_text: the file body
    :return: the columns
    """
    import json

    d = json.loads(benchmark_text)
    mem = d.get("infer_peak_reserved_GB")
    if mem is None:
        mem = d.get("infer_peak_alloc_GB")
    return {
        "rtf": _round(d.get("rtf"), 4),
        "am_rtf": _round(d.get("am_rtf"), 4),
        "search_rtf": _round(d.get("search_rtf"), 4),
        "utts_per_s": _round(d.get("utts_per_s"), 2),
        "infer_mem_GB": _round(mem, 2),
        "mem_at_s": _round(d.get("mem_driver_audio_s"), 1),
    }


def wer_group(label: str) -> str:
    """
    :param label: a recognition label
    :return: its group, greedy, beam, joint_ctc or lm, by the name parts the evaluations use
    """
    if "lm" in label or "4gram" in label:
        return "lm"
    if "ctc" in label:
        return "joint_ctc"
    if "beam" in label:
        return "beam"
    return "greedy"


def resolve_wer_labels(wers: Dict[str, float], names: Dict[str, str]) -> Dict[str, float]:
    """
    Renames recognition labels by their checkpoint names, merging labels that name the same checkpoint.

    :param wers: the WER per label
    :param names: the checkpoint name per label
    :return: the lowest WER per checkpoint name
    """
    out: Dict[str, float] = {}
    for label, wer in wers.items():
        name = resolve_label(label, names)
        out[name] = min(wer, out[name]) if name in out else wer
    return out


def best_wer_by_group(wers: Dict[str, float]) -> Dict[str, Tuple[str, float]]:
    """
    :param wers: the WER per label
    :return: per group the label with the lowest WER and that WER, the minimum the best-by-WER selection takes
    """
    best: Dict[str, Tuple[str, float]] = {}
    for label, wer in wers.items():
        group = wer_group(label)
        if group not in best or wer < best[group][1]:
            best[group] = (label, wer)
    return best


def format_metrics_markdown(m: Dict[str, Any]) -> str:
    """
    :param m: the metrics of one model
    :return: the metrics as a markdown report
    """
    rows: List[str] = []

    def row(label, value):
        rows.append(f"| {label} | {value} |")

    def has(*keys):
        return all(m.get(k) is not None for k in keys)

    def wer_row(title, key):
        if has(key):
            label = f" ({m[key + '_label']})" if m.get(key + "_label") else ""
            row(title, f"{m[key]} %{label}")

    wer_row("WER (greedy)", "wer")
    wer_row("WER (beam)", "wer_beam")
    wer_row("WER (joint CTC)", "wer_joint_ctc")
    if has("sub", "del", "ins"):
        row("Sub / Del / Ins", f"{m['sub']} / {m['del']} / {m['ins']} %")
    if has("rtf"):
        row("RTF (single-stream)", m["rtf"])
    if has("am_rtf", "search_rtf"):
        row("RTF encoder / search", f"{m['am_rtf']} / {m['search_rtf']}")
    if has("utts_per_s"):
        row("Throughput", f"{m['utts_per_s']} utt/s")
    if has("infer_mem_GB"):
        at = (
            f" (at {m['mem_at_s']} s utterance)"
            if m.get("mem_at_s") is not None
            else ""
        )
        row("Inference peak memory", f"{m['infer_mem_GB']} GB{at}")
    if has("params_M"):
        row("Parameters", f"{m['params_M']} M")
    if m.get("params_by_component"):
        row(
            "Parameters by component",
            ", ".join(
                f"{k} {round(v, 1)}M" for k, v in m["params_by_component"].items()
            ),
        )
    if has("ckpt_MB"):
        row("Checkpoint size", f"{m['ckpt_MB']} MB")
    if has("train_time_h"):
        gpu = f" ({m['gpu_h']} GPU-h)" if m.get("gpu_h") is not None else ""
        row("Training time", f"{m['train_time_h']} h{gpu}")
    if has("peak_train_mem_GB"):
        pool = f" (of which {m['graph_pool_GB']} GB graph pool)" if m.get("graph_pool_GB") else ""
        row("Peak training memory", f"{m['peak_train_mem_GB']} GB{pool}")
    if has("epochs_done"):
        steps = f" / {m['total_steps']:,} steps" if m.get("total_steps") else ""
        row("Epochs", f"{m['epochs_done']}{steps}")
    if m.get("train_device"):
        row("Training device", m["train_device"])
    if has("frame_rate"):
        row("Frame rate", f"{m['frame_rate']} Hz")
    if m.get("delay_s") is not None:
        row("Delay", f"{m['delay_s']} s")
    if m.get("spm"):
        row("SPM vocab", m["spm"])

    header = f"# {m.get('name', 'model')}"
    if m.get("corpus"):
        header += f" ({m['corpus']})"
    table = "| Metric | Value |\n| --- | --- |\n" + "\n".join(rows)
    report = f"{header}\n\n{table}\n"
    if m.get("wer_all"):
        all_rows = "\n".join(
            f"| {label} | {wer} |" for label, wer in sorted(m["wer_all"].items(), key=lambda kv: kv[1])
        )
        report += f"\n## All WERs\n\n| Label | WER |\n| --- | --- |\n{all_rows}\n"
    if m.get("wer_by_source"):
        by_source = m["wer_by_source"]
        sources = sorted({source for parts in by_source.values() for source in parts})
        order = sorted(by_source, key=lambda label: (m.get("wer_all", {}).get(label, float("inf")), label))
        source_rows = "\n".join(
            f"| {label} | " + " | ".join(_cell(by_source[label].get(source)) for source in sources) + " |"
            for label in order
        )
        header = "| Label | " + " | ".join(sources) + " |\n| --- | " + " | ".join("---" for _ in sources) + " |\n"
        report += f"\n## WER by source\n\n{header}{source_rows}\n"
    return report


def _cell(value) -> str:
    return "" if value is None else str(value)


class ModelMetricsJob(Job):
    """Assemble one parse-only metrics row for a finished training and its recognition."""

    __sis_hash_exclude__ = {"label_names": None, "source_wer_vars": None}

    def __init__(
        self,
        *,
        learning_rates,
        checkpoint,
        wer_vars,
        model_meta,
        num_gpus=1,
        benchmark_json=None,
        breakdown_vars=None,
        label_names=None,
        source_wer_vars=None,
    ):
        self.learning_rates = learning_rates
        self.checkpoint = checkpoint
        self.wer_vars = dict(wer_vars)
        self.breakdown_vars = {label: dict(parts) for label, parts in (breakdown_vars or {}).items()}
        self.source_wer_vars = {label: dict(parts) for label, parts in (source_wer_vars or {}).items()} or None
        self.label_names = label_names
        self.model_meta = dict(model_meta)
        self.num_gpus = num_gpus
        self.benchmark_json = benchmark_json
        self.out_metrics = self.output_var("metrics")
        self.out_report = self.output_path("metrics.md")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        lr_path = self.learning_rates.get_path()
        with open(lr_path) as f:
            timing = parse_training_time(f.read())

        job_dir = os.path.dirname(os.path.dirname(lr_path))
        log_text = rank_zero_log_text(job_dir)
        for extra in glob.glob(os.path.join(job_dir, "work", "returnn.log")):
            try:
                with open(extra, errors="ignore") as f:
                    log_text += f.read()
            except OSError:
                continue
        mem = parse_peak_mem_gb(log_text)

        params_total, by_comp = count_params(self.checkpoint.get_path())
        ckpt_mb = os.path.getsize(self.checkpoint.get_path()) / _UNIT["M"]
        names = resolve_names(self.label_names)
        wer_all = resolve_wer_labels(
            {label: float(v.get() if hasattr(v, "get") else v) for label, v in self.wer_vars.items()}, names
        )
        wer: Dict[str, Any] = {"wer_all": dict(sorted(wer_all.items(), key=lambda kv: kv[1]))}
        for group, (label, value) in best_wer_by_group(wer_all).items():
            key = "wer" if group == "greedy" else f"wer_{group}"
            wer[key] = value
            wer[f"{key}_label"] = label
        breakdown = {resolve_label(label, names): parts for label, parts in self.breakdown_vars.items()}
        for part, v in breakdown.get(wer.get("wer_label"), {}).items():
            wer[part] = v.get() if hasattr(v, "get") else v
        if self.source_wer_vars:
            by_source: Dict[str, Dict[str, float]] = {}
            for label, parts in self.source_wer_vars.items():
                values = {src: (v.get() if hasattr(v, "get") else v) for src, v in parts.items()}
                values = {src: v for src, v in values.items() if v is not None}
                name = resolve_label(label, names)
                if name not in by_source or wer_all.get(name) == float(self.wer_vars[label].get()):
                    by_source[name] = values
            wer["wer_by_source"] = by_source

        bench = {}
        if self.benchmark_json is not None:
            with open(self.benchmark_json.get_path()) as f:
                bench = parse_benchmark(f.read())

        peak_train = mem["peak_usage_gb"]
        if peak_train is None:
            peak_train = mem["peak_reserved_gb"] if mem["peak_reserved_gb"] is not None else mem["peak_alloc_gb"]

        metrics = {
            **self.model_meta,
            **wer,
            **bench,
            "params_total": params_total,
            "params_by_component": {k: round(v / 1e6, 3) for k, v in by_comp.items()},
            "params_M": round(params_total / 1e6, 2),
            "ckpt_MB": round(ckpt_mb, 1),
            "train_time_h": round(timing["train_time_s"] / 3600.0, 2),
            "gpu_h": round(timing["train_time_s"] / 3600.0 * self.num_gpus, 2),
            "epochs_done": timing["epochs_done"],
            "total_steps": timing["total_steps"],
            "train_device": timing["device"],
            "peak_train_mem_GB": None if peak_train is None else round(peak_train, 2),
            "graph_pool_GB": _round(mem["graph_pool_gb"], 2),
        }
        self.out_metrics.set(metrics)
        with open(self.out_report.get_path(), "w") as f:
            f.write(format_metrics_markdown(metrics))


_TABLE_COLS = [
    "name",
    "corpus",
    "wer",
    "wer_label",
    "wer_beam",
    "wer_beam_label",
    "wer_joint_ctc",
    "wer_joint_ctc_label",
    "sub",
    "del",
    "ins",
    "rtf",
    "am_rtf",
    "search_rtf",
    "utts_per_s",
    "infer_mem_GB",
    "mem_at_s",
    "params_M",
    "ckpt_MB",
    "train_time_h",
    "gpu_h",
    "peak_train_mem_GB",
    "graph_pool_GB",
    "frame_rate",
    "delay_s",
    "spm",
    "epochs",
]


class CollectMetricsTableJob(Job):
    """Aggregate per-model metrics dicts into one markdown + tsv comparison table."""

    def __init__(self, *, metrics_vars, columns=None):
        self.metrics_vars = list(metrics_vars)
        self.columns = list(columns) if columns else list(_TABLE_COLS)
        self.out_table = self.output_path("table.md")
        self.out_tsv = self.output_path("table.tsv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rows = [v.get() for v in self.metrics_vars]
        cols = self.columns

        def cell(row, col):
            x = row.get(col, "")
            return "" if x is None else str(x)

        with open(self.out_tsv.get_path(), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(cell(r, c) for c in cols) + "\n")
        with open(self.out_table.get_path(), "w") as f:
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join("---" for _ in cols) + " |\n")
            for r in rows:
                f.write("| " + " | ".join(cell(r, c) for c in cols) + " |\n")
