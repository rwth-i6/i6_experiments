"""Frozen, dev-selected comparisons; analysis only, no model inference.

Selection and evaluation consume separate snapshots. Paired bootstrap resamples
recordings/chapters where identifiable, never pools overlapping corpora, and
recomputes the better *aggregate* standalone baseline in every replicate.
"""

import ast
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sisyphus import Job, Task

DEV = ("librispeech-dev-other", "loquacious-dev.short")
GROUPS = ("qwen-granite", "qwen-granite-1.7b", "qwen-parakeet-rnnt", "qwen-parakeet-rnnt-1.7b",
          "qwen-speech-0.6b", "qwen-speech-1.7b", "qwen-parakeet-rnnt-speech-1.7b")
NORMALIZER = "48219c6028db0517d704600d92f31edfc96e8c23"


def read_json(path):
    with open(str(path), encoding="utf-8") as src:
        return json.load(src)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_version():
    from . import audio_provenance
    return hashlib.sha256(Path(__file__).read_bytes() + Path(audio_provenance.__file__).read_bytes()).hexdigest()


def info_parameters(job_dir):
    """Parse trusted Sisyphus metadata without loading pickles or evaluating code."""
    result = {}
    path = Path(job_dir) / "info"
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if not line.startswith("PARAMETER: "):
            continue
        key, value = line[len("PARAMETER: "):].split(": ", 1)
        # Subclass **kwargs contain repr(Path); replace only this known syntax.
        literal = re.sub(r"<Path ([^>]+)>", lambda m: repr(m[1]), value)
        try:
            result[key] = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            result[key] = value
    return result


def parameters_flat(job_dir):
    result = info_parameters(job_dir)
    kwargs = result.pop("kwargs", {})
    if isinstance(kwargs, dict):
        result = dict(kwargs, **result)
    return result


def provenance(job_dir, depth=0, seen=None):
    seen = set() if seen is None else seen
    job_dir = Path(job_dir).resolve()
    if job_dir in seen or depth > 4:
        return []
    seen.add(job_dir)
    params = parameters_flat(job_dir)
    resources = {}
    usage = job_dir / "usage.run.1"
    if usage.exists():
        try:
            data = ast.literal_eval(usage.read_text())
            resources = dict(host_peak_rss_gb=data.get("max", {}).get("rss"),
                             requested=data.get("requested_resources"), host=data.get("host"),
                             gpu_peak_allocated_gb=None)
        except (ValueError, SyntaxError):
            pass
    files = {}
    for key in ("config", "rasr_config", "prior", "vocabulary", "gate"):
        value = params.get(key)
        if isinstance(value, str) and Path(value).is_file():
            files[key] = dict(path=value, sha256=sha256(value))
    repositories = {}
    for key in ("repo", "speech_repo", "granite_repo", "qwen_repo", "auxiliary_repo"):
        value = params.get(key)
        if isinstance(value, str) and "/output/" in value:
            repositories[key] = parameters_flat(Path(value).parent.parent)
    row = dict(job=str(job_dir), job_type=job_dir.name.split(".")[0], parameters=params, repositories=repositories,
               info_sha256=sha256(job_dir / "info") if (job_dir / "info").exists() else None,
               files=files, resources=resources)
    rows = [row]
    for key in ("candidates", "generation_runtime", "scoring_runtime", "generation", "lm_scoring", "hypotheses"):
        value = params.get(key)
        if isinstance(value, str) and "/output/" in value:
            rows += provenance(Path(value).parent.parent, depth + 1, seen)
    return rows


def family(row):
    model, auxiliary, method = row["model"], row["lm"], row["decoding"]
    if row["condition"] == "speech-nbest-scores":
        return None  # LM-scoring stage, not a recognition/scoring condition.
    if model not in {"Granite CTC", "Parakeet RNN-T", "Speech Qwen 0.6B", "Speech Qwen 1.7B"}:
        return None
    if auxiliary == "None" and (row["condition"].startswith(("native", "zero-", "first-pass-"))):
        method = "standalone"
    elif method.startswith("1-pass") and auxiliary.startswith(("Text Qwen", "Speech Qwen")):
        method = "one-pass"
    elif method.startswith("2-pass") and auxiliary.startswith(("Text Qwen", "Speech Qwen")):
        method = "two-pass"
    else:
        return None
    if "earnings-prune" in row["condition"]:
        return None  # Explicit test-specific tuning is not a transferable candidate.
    return " | ".join((model, auxiliary, method))


def scan_exports(output_root, alias_root):
    """Inventory existing exports/aliases, including unfinished expected dev runs.

    Returns serializable records, not Job objects; never reconstructs old grids.
    Snapshot hashes below include score/runtime bytes and provenance metadata.
    """
    from .result_collection import describe
    output_root, alias_root = Path(output_root), Path(alias_root)
    rows = []
    for group in GROUPS:
        bases = [output_root / "open-asr" / group, alias_root / "open-asr" / group]
        conditions = sorted({p.name for b in bases if b.is_dir() for p in b.iterdir() if p.is_dir()})
        for condition in conditions:
            metadata = dict(group=group, condition=condition)
            metadata["model"], metadata["lm"], metadata["decoding"], metadata["lexicon"] = describe(metadata)
            f = family(metadata)
            if f is None:
                continue
            dirs = [b / condition for b in bases]
            datasets = sorted({p.name for d in dirs if d.is_dir() for p in d.iterdir() if p.is_dir()})
            for dataset in datasets:
                if dataset not in DEV and not (dataset.startswith(("librispeech-test-", "loquacious-test"))
                                               or dataset in {"ami-cleaned", "earnings22-cleaned", "gigaspeech-cleaned",
                                                              "spgispeech", "voxpopuli-cleaned", "monsoon-en-in"}):
                    continue
                export = dirs[0] / dataset
                alias = dirs[1] / dataset
                score = export / "scoring"
                if not score.exists():
                    score = alias / "scoring" / "output" / "summary.json"
                if not score.exists():
                    if dataset in DEV and ((alias / "scoring").exists() or (alias / "scoring").is_symlink()):
                        rows.append(dict(metadata, dataset=dataset, family=f, status="pending", score=str(score)))
                    continue
                score = score.resolve()
                try:
                    summary = read_json(score)
                except (OSError, ValueError):
                    rows.append(dict(metadata, dataset=dataset, family=f, status="invalid", score=str(score)))
                    continue
                score_params = parameters_flat(score.parent.parent)
                ref = score_params.get("references")
                runtime = export / "runtime"
                if not runtime.exists():
                    # Some Loquacious subset exports omit runtime. Do not claim
                    # subset timing; use parent decoding only to enforce budget.
                    parent = dataset.rsplit(".", 1)[0] if dataset.startswith("loquacious-") and "." in dataset else None
                    if parent:
                        runtime = dirs[0] / parent / "runtime"
                runtime_path = str(runtime.resolve()) if runtime.exists() else None
                hypothesis = export / "hypotheses"
                if not hypothesis.exists() and isinstance(score_params.get("hypotheses"), str):
                    hypothesis = Path(score_params["hypotheses"])
                if not hypothesis.exists():
                    continue
                stages = provenance(hypothesis.resolve().parent.parent)
                if runtime_path:
                    stages += provenance(Path(runtime_path).parent.parent, seen={Path(s["job"]) for s in stages})
                dataset_paths = sorted({str(Path(s["parameters"]["dataset"]).resolve()) for s in stages
                                        if isinstance(s["parameters"].get("dataset"), str)})
                rows.append(dict(metadata, dataset=dataset, family=f, status="complete", score=str(score),
                    score_sha256=sha256(score), segments=str(score.parent / "segments.jsonl"),
                    runtime=runtime_path, runtime_sha256=sha256(runtime_path) if runtime_path else None,
                    runtime_is_parent=(dataset.startswith("loquacious-") and runtime_path is not None
                                       and not (export / "runtime").exists()),
                    references=ref, reference_sha256=sha256(ref) if ref and Path(ref).exists() else None,
                    hypotheses=str(hypothesis.resolve()), dataset_paths=dataset_paths, stages=stages,
                    normalizer=summary.get("normalizer_revision")))
    return rows


class SnapshotControlledResultsJob(Job):
    def __init__(self, records, *, partition, code_version):
        self.records, self.partition, self.code_version = records, partition, code_version
        if any((r["dataset"] in DEV) != (partition == "dev") for r in records):
            raise ValueError("Dev and evaluation evidence must be separate inputs")
        self.out_snapshot = self.output_path("snapshot.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        result = []
        for record in self.records:
            row = dict(record)
            if row["status"] == "complete":
                for field in ("score", "runtime"):
                    if row.get(field) and sha256(row[field]) != row[field + "_sha256"]:
                        raise ValueError("Evidence changed since graph construction; reload the snapshot")
                row["summary"] = read_json(row["score"])
                row["timing"] = read_json(row["runtime"]) if row.get("runtime") else {}
            result.append(row)
        with open(self.out_snapshot.get_path(), "w") as dst:
            json.dump(dict(partition=self.partition, normalizer=NORMALIZER, records=result), dst, indent=2)


def select_protocol(snapshot, *, max_rtf=1.5):
    if snapshot["partition"] != "dev" or any(r["dataset"] not in DEV for r in snapshot["records"]):
        raise ValueError("Selection may only see development data")
    rows = snapshot["records"]
    anchors = {}
    for row in rows:
        if row["status"] == "complete" and row["condition"] == "native" and row["lm"] == "None":
            key = row["model"], row["dataset"]
            value = row["summary"]["wer_percent"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid native development anchor")
            if key in anchors and abs(anchors[key] - value) > 1e-6:
                raise ValueError("Conflicting native development anchors")
            anchors[key] = value
    variants = defaultdict(dict)
    for row in rows:
        variants[(row["family"], row["group"], row["condition"])][row["dataset"]] = row
    eligible, audit = defaultdict(list), []
    for (f, group, condition), dev in sorted(variants.items()):
        reason = None
        if "prune-threshold" in condition or "prune-beam" in condition:
            reason = "staged pruning diagnostic, not a finalized configuration"
        elif set(dev) != set(DEV):
            reason = "not present on both development corpora"
        elif any(r["status"] != "complete" for r in dev.values()):
            reason = "pending development results"
        elif any(r["normalizer"] != NORMALIZER for r in dev.values()):
            reason = "normalizer mismatch"
        elif any(not r.get("reference_sha256") for r in dev.values()):
            reason = "missing reference provenance"
        elif any(not isinstance(r["summary"].get("wer_percent"), (int, float))
                 or not math.isfinite(r["summary"]["wer_percent"]) or r["summary"]["wer_percent"] < 0 for r in dev.values()):
            reason = "invalid development WER"
        elif any((r["model"], ds) not in anchors for ds, r in dev.items()):
            reason = "missing native anchor"
        else:
            rtfs = [r["timing"].get("rtf") for r in dev.values()]
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in rtfs):
                reason = "missing/invalid development timing"
        entry = dict(family=f, group=group, condition=condition, reason=reason)
        if reason is None:
            entry.update(objective=float(np.mean([r["summary"]["wer_percent"] / anchors[r["model"], ds]
                                                   for ds, r in dev.items()])),
                         max_dev_rtf=max(rtfs), dev=dev)
            eligible[f].append(entry)
        audit.append(entry)
    selected = {}
    for f, choices in eligible.items():
        for operating_point in ("fast", "unrestricted"):
            allowed = [c for c in choices if operating_point == "unrestricted" or c["max_dev_rtf"] < max_rtf]
            if allowed:
                selected[f + " | " + operating_point] = min(allowed,
                    key=lambda c: (c["objective"], c["max_dev_rtf"], c["group"], c["condition"]))
    pending = [r for r in audit if r["reason"] == "pending development results"]
    return dict(schema=1, ready=not pending, dev_datasets=list(DEV), max_rtf=max_rtf,
                objective="Equal-weight mean WER/native-WER over the two dev corpora; ties: runtime then id",
                evaluation_status="Exploratory: existing test sets have already been inspected",
                prior_transfer="Loquacious recipe for Loquacious; LibriSpeech recipe for all other corpora",
                selected=selected, candidate_audit=audit)


class SelectControlledConfigsJob(Job):
    def __init__(self, dev_snapshot, *, max_rtf=1.5, code_version):
        self.dev_snapshot, self.max_rtf, self.code_version = dev_snapshot, max_rtf, code_version
        self.out_protocol = self.output_path("selection.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        protocol = select_protocol(read_json(self.dev_snapshot.get_path()), max_rtf=self.max_rtf)
        protocol["dev_snapshot"] = self.dev_snapshot.get_path()
        protocol["dev_snapshot_sha256"] = sha256(self.dev_snapshot.get_path())
        protocol["selection_code_version"] = self.code_version
        with open(self.out_protocol.get_path(), "w") as dst:
            json.dump(protocol, dst, indent=2)


class CollectSelectedEvaluationJob(Job):
    """Collect declared results only after their actual scoring jobs finish."""
    def __init__(self, entries, *, code_version):
        self.entries, self.code_version = entries, code_version
        self.out_snapshot = self.output_path("evaluation.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        rows = []
        for entry in self.entries:
            row = dict(entry)
            for field in ("score", "segments", "hypotheses", "runtime", "references"):
                if row.get(field) is not None:
                    value = row[field]
                    row[field] = value.get_path() if hasattr(value, "get_path") else str(value)
            score = read_json(row["score"])
            stages = provenance(Path(row["hypotheses"]).resolve().parent.parent)
            if row.get("runtime"):
                stages += provenance(Path(row["runtime"]).resolve().parent.parent, seen={Path(s["job"]) for s in stages})
            row.update(status="complete", summary=score, timing=read_json(row["runtime"]) if row.get("runtime") else {},
                score_sha256=sha256(row["score"]), normalizer=score.get("normalizer_revision"),
                reference_sha256=sha256(row["references"]), stages=stages,
                dataset_paths=sorted({str(Path(s["parameters"]["dataset"]).resolve()) for s in stages
                                     if isinstance(s["parameters"].get("dataset"), str)}))
            rows.append(row)
        with open(self.out_snapshot.get_path(), "w") as dst:
            json.dump(dict(partition="evaluation", records=rows), dst, indent=2)


def cluster_id(dataset, seq_id):
    """Prespecified, reviewable source-id grouping; unknown ids stay utterances."""
    if dataset.startswith("librispeech-") or (dataset.startswith("loquacious-") and re.fullmatch(r"\d+-\d+-\d+", seq_id)):
        match = re.fullmatch(r"(\d+-\d+)-\d+", seq_id)
        if match:
            return "chapter:" + match[1]
    if dataset == "ami-cleaned":
        match = re.match(r"AMI_([^_]+)_", seq_id)
        if match:
            return "meeting:" + match[1]
    if dataset == "gigaspeech-cleaned":
        match = re.fullmatch(r"(.+)_S\d+", seq_id)
        if match:
            return "recording:" + match[1]
    if dataset == "spgispeech" and "/" in seq_id:
        return "recording:" + seq_id.rsplit("/", 1)[0]
    if dataset == "voxpopuli-cleaned" and "-PLENARY-" in seq_id:
        return "session:" + seq_id.split("_", 1)[0]
    if dataset == "monsoon-en-in" and re.fullmatch(r"\d+_\d+_\d+_\d+_\d+", seq_id):
        return "recording:" + seq_id.split("_", 1)[0]
    if dataset.startswith("loquacious-"):
        match = re.fullmatch(r"(.+)-\d{5}-\d{8}-\d{8}\.wav", seq_id)
        if match:
            return "recording:" + match[1]
    return "scoring-unit:" + seq_id


def paired_bootstrap(counts, baseline_indices, *, seed=20261103, samples=10000):
    """Counts [cluster, system, (errors, reference words)]; paired ratio WER.

    Each system retains its own denominator because compound-equivalence
    normalization can change reference word counts slightly by hypothesis.
    Baselines are dataset-level standalones, not an utterance-selection oracle.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or counts.shape[2] != 2 or samples < 100 or counts.shape[0] < 1:
        raise ValueError("Invalid bootstrap input")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(0)[:, 1] <= 0).any():
        raise ValueError("Invalid scoring counts")
    rng = np.random.default_rng(seed)
    n, systems, _ = counts.shape
    point = 100 * counts.sum(0)[:, 0] / counts.sum(0)[:, 1]
    wer_samples = np.empty((samples, systems))
    for start in range(0, samples, 32):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(32, samples - start))
        totals = weights @ counts.reshape(n, -1)
        totals = totals.reshape(len(weights), systems, 2)
        if (totals[:, :, 1] <= 0).any():
            raise ValueError("Bootstrap sampled only zero-reference units")
        wer_samples[start:start + len(weights)] = 100 * totals[:, :, 0] / totals[:, :, 1]
    result = []
    for i in range(systems):
        baselines = baseline_indices[i]
        if any(j < 0 or j >= systems for j in baselines):
            raise ValueError("Invalid standalone baseline index")
        row = dict(wer=float(point[i]), wer_ci=np.quantile(wer_samples[:, i], [.025, .975]).tolist())
        if baselines:
            deltas = wer_samples[:, i] - wer_samples[:, baselines].min(1)
            row.update(delta=float(point[i] - point[baselines].min()),
                       delta_ci=np.quantile(deltas, [.025, .975]).tolist(),
                       baseline_indices=baselines, baselines=[dict(index=j, delta=float(point[i] - point[j]),
                         delta_ci=np.quantile(wer_samples[:, i] - wer_samples[:, j], [.025, .975]).tolist())
                         for j in baselines])
        result.append(row)
    return result


def compare_dataset(dataset, records, *, samples=10000, seed=20261103, max_rtf=1.5):
    if not records:
        return dict(dataset=dataset, rows=[], status="no exact selected configuration available")
    refs = {r["reference_sha256"] for r in records}
    if None in refs or len(refs) != 1:
        raise ValueError("Selected systems have different raw references")
    from .audio_provenance import audio_origins
    origins = {audio_origins(r) for r in records if r["dataset_paths"]}
    if len(origins) > 1:
        raise ValueError("Selected systems have different prepared audio datasets")
    all_counts, ids = [], None
    for record in records:
        if record["normalizer"] != NORMALIZER:
            raise ValueError("Normalizer mismatch")
        segments = {}
        with open(record["segments"]) as src:
            for line in src:
                row = json.loads(line)
                if row["id"] in segments:
                    raise ValueError("Duplicate scoring unit")
                segments[row["id"]] = row["counts"]
        if ids is None:
            ids = sorted(segments)
        if set(ids) != set(segments):
            raise ValueError("Paired scoring coverage differs")
        expected = record["summary"]
        for key in ("ins", "del", "sub", "total", "ref_len"):
            if sum(c[key] for c in segments.values()) != expected[key]:
                raise ValueError("Segment counts differ from scoring summary")
        all_counts.append(segments)
    clusters = sorted({cluster_id(dataset, i) for i in ids})
    cluster_index = {c: i for i, c in enumerate(clusters)}
    counts = np.zeros((len(clusters), len(records), 2))
    for unit in ids:
        for j, segments in enumerate(all_counts):
            row = segments[unit]
            counts[cluster_index[cluster_id(dataset, unit)], j] += row["total"], row["ref_len"]
    standalone = {r["model"]: i for i, r in enumerate(records) if r["lm"] == "None"}
    baselines = []
    for r in records:
        candidates = [standalone[m] for m in (r["model"], r["lm"]) if m in standalone]
        baselines.append(candidates if r["lm"] != "None" else [])
    stats = paired_bootstrap(counts, baselines, samples=samples, seed=seed)
    result = []
    for r, stat in zip(records, stats):
        actual_rtf = r["timing"].get("rtf")
        missing_baseline = r["lm"] != "None" and (r["model"] not in standalone
            or (r["lm"].startswith("Speech Qwen") and r["lm"] not in standalone))
        # Do not silently relabel an ASR-only comparison as beating the better
        # component when the Speech-Qwen standalone control is missing.
        if missing_baseline:
            for key in ("delta", "delta_ci", "baselines", "baseline_indices"):
                stat.pop(key, None)
        result.append(dict(r, **stat, missing_baseline=missing_baseline,
                          meets_eval_budget=isinstance(actual_rtf, (int, float)) and actual_rtf < max_rtf,
                          segments_sha256=sha256(r["segments"])))
    return dict(dataset=dataset, rows=result, n_scoring_units=len(ids), n_clusters=len(clusters),
                cluster_ids=clusters, cluster_policy="Identifier-derived source recording/session/chapter; otherwise scoring unit",
                uncertainty_warning="Few independent clusters; interval is fragile" if len(clusters) < 30 else None,
                status="exploratory, unadjusted paired 95% percentile cluster-bootstrap intervals")


class ControlledComparisonJob(Job):
    def __init__(self, protocol, evaluation_snapshot, *, bootstrap_samples=10000, seed=20261103,
                 operating_points=("fast", "unrestricted"), code_version):
        self.protocol, self.evaluation_snapshot = protocol, evaluation_snapshot
        self.bootstrap_samples, self.seed, self.code_version = bootstrap_samples, seed, code_version
        self.operating_points = operating_points
        self.out_report = self.output_path("comparison.md")
        self.out_results = self.output_path("comparison.json")
        self.out_tsv = self.output_path("comparison.tsv")
        self.out_missing = self.output_path("missing-evaluations.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=8))

    def run(self):
        protocol = read_json(self.protocol.get_path())
        snapshot = read_json(self.evaluation_snapshot.get_path())
        if snapshot["partition"] != "evaluation":
            raise ValueError("Expected separate evaluation evidence")
        by_key = {(r["group"], r["condition"], r["dataset"]): r for r in snapshot["records"] if r["status"] == "complete"}
        datasets = sorted({r["dataset"] for r in snapshot["records"]})
        results, missing = {}, []
        for point in self.operating_points:
            selected = [c for k, c in protocol["selected"].items() if k.endswith(" | " + point)]
            for dataset in datasets:
                records = []
                for choice in selected:
                    key = choice["group"], choice["condition"], dataset
                    row = by_key.get(key)
                    if row is None:
                        missing.append(dict(operating_point=point, dataset=dataset, family=choice["family"],
                                            group=choice["group"], condition=choice["condition"]))
                    else:
                        records.append(row)
                results[point + "/" + dataset] = compare_dataset(dataset, records, samples=self.bootstrap_samples,
                                                                 seed=self.seed, max_rtf=protocol["max_rtf"])
        report = render_report(protocol, results, missing)
        with open(self.out_report.get_path(), "w") as dst:
            dst.write(report)
        with open(self.out_results.get_path(), "w") as dst:
            json.dump(dict(protocol_sha256=sha256(self.protocol.get_path()), seed=self.seed,
                           bootstrap_samples=self.bootstrap_samples, comparisons=results), dst, indent=2)
        with open(self.out_missing.get_path(), "w") as dst:
            json.dump(missing, dst, indent=2)
        fields = ["operating_point", "dataset", "model", "lm", "decoding", "lexicon", "condition", "wer", "delta", "delta_ci",
                  "ins", "del", "sub", "ref_len", "rtf", "budget_ok", "clusters", "source"]
        with open(self.out_tsv.get_path(), "w") as dst:
            writer = csv.DictWriter(dst, fields, delimiter="\t")
            writer.writeheader()
            for key, comparison in results.items():
                for row in comparison["rows"]:
                    writer.writerow(dict(operating_point=key.split("/", 1)[0], dataset=comparison["dataset"],
                        **{f: row.get(f) for f in ("model", "lm", "decoding", "lexicon", "condition", "wer", "delta", "delta_ci")},
                        **{f: row["summary"].get(f) for f in ("ins", "del", "sub", "ref_len")},
                        rtf=row["timing"].get("rtf"), budget_ok=row["meets_eval_budget"],
                        clusters=comparison["n_clusters"], source=row["score"]))


def render_report(protocol, results, missing):
    lines = ["# Controlled pretrained-ASR comparisons", "",
             "Development-selected configurations; no test-set configuration selection. Existing test results are exploratory.",
             f"Fast = RTF <{protocol['max_rtf']} on BOTH dev corpora, not a guarantee on evaluation data. Unrestricted = accuracy reference.",
             "Objective: " + protocol["objective"],
             "Selection complete: " + str(protocol["ready"]) + ". Missing exact evaluations: " + str(len(missing)) + ".", "",
             "## Frozen development choices", "",
             "| Family / point | Condition | Dev-other WER | Dev.short WER | Max dev RTF |", "|---|---|---:|---:|---:|"]
    for label, choice in sorted(protocol["selected"].items()):
        lines.append(f"| {label.replace(' | ', ' / ')} | {choice['group']}/{choice['condition']} | "
                     f"{choice['dev'][DEV[0]]['summary']['wer_percent']:.3f} | "
                     f"{choice['dev'][DEV[1]]['summary']['wer_percent']:.3f} | {choice['max_dev_rtf']:.3f} |")
    lines += ["", "Intervals are paired 95% cluster-bootstrap percentile intervals, unadjusted for multiple comparisons.",
              "ΔWER is system minus the better aggregate standalone component; baseline minimum is recomputed within each replicate.",
              "Reference word counts remain system-specific for compound-equivalence scoring. Parent/subsets are never pooled.",
              "Clusters follow declared ID-derived recording/session/chapter rules; unknown origins remain scoring units.",
              "Host RSS is not GPU memory. GPU request is an allocation, not measured peak use; unknown GPU peaks remain unknown.", ""]
    lines += ["Reused results are not necessarily hardware-matched. Hosts and requested resources are retained in JSON; "
              "strict speed claims require a matched-hardware benchmark. Parent RTF is marked, not measured separately for subsets.", ""]
    for key, comparison in sorted(results.items()):
        lines += ["## " + key, ""]
        if not comparison["rows"]:
            lines += [comparison["status"], ""]
            continue
        lines += [f"{comparison['n_scoring_units']} scoring units; {comparison['n_clusters']} clusters."]
        if comparison["uncertainty_warning"]:
            lines += [comparison["uncertainty_warning"]]
        lines += ["", "| ASR | Auxiliary | Method | WER | I / D / S | ΔWER [95% CI] | RTF | Budget | Host RSS / GPU request (GB) |",
                  "|---|---|---|---:|---:|---:|---:|---|---|"]
        for r in comparison["rows"]:
            summary = r["summary"]
            delta = "—" if "delta" not in r else f"{r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}]"
            rt = r["timing"].get("rtf")
            resources = [s["resources"] for s in r["stages"] if s["resources"]]
            rss = [s["host_peak_rss_gb"] for s in resources if s.get("host_peak_rss_gb") is not None]
            gpu = [(s.get("requested") or {}).get("gpu_mem") for s in resources if (s.get("requested") or {}).get("gpu_mem") is not None]
            memory = f"{max(rss):.2f}" if rss else "—"
            memory += " / " + (str(max(gpu)) if gpu else "—")
            lines.append(f"| {r['model']} | {r['lm']} | {r['decoding']} | {r['wer']:.3f} | "
                         f"{summary['ins']} / {summary['del']} / {summary['sub']} | {delta} | "
                         + ((f"{rt:.3f}" + (" (parent)" if r.get("runtime_is_parent") else "")) if rt is not None else "—")
                         + f" | {'yes' if r['meets_eval_budget'] else 'no/unknown'} | {memory} |")
        lines += [""]
    lines += ["## Missing selected evaluations", "", "No nearby/default/test-best configuration is substituted for a missing selected one.", ""]
    for row in missing:
        lines.append(f"- {row['operating_point']} / {row['dataset']}: {row['family']} → {row['group']}/{row['condition']}")
    return "\n".join(lines) + "\n"
