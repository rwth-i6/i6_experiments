"""Read-only diagnostics and preflight jobs for the second OOV prototype."""

import configparser
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter

from i6_core.util import uopen
from sisyphus import Job, Task


TRACE = re.compile(
    r"<TracebackItem\(lemma='(.*)', am_score=([^,]*), lm_score=([^,]*), start_time=([^,]*), end_time=([^,)]+)"
)


def read_tracebacks(path):
    """Read the selected search traces, not beam-wide RASR counters.

    The binding's repr does not escape apostrophes in lemma names, so literal_eval
    is not appropriate. Match the fixed field delimiters instead.
    """
    segment = None
    in_search = False
    with uopen(path, "rt") as src:
        for line in src:
            if line.startswith("Recognize sequence "):
                if segment is not None:
                    yield segment
                segment = dict(tag=line.strip()[len("Recognize sequence ") :].strip("'\""), items=[])
                in_search = False
            elif segment is not None:
                if "Ground truth:" in line:
                    segment["reference"] = line.split("Ground truth:", 1)[1].strip().strip('"')
                elif "Recognized:" in line:
                    segment["hypothesis"] = line.split("Recognized:", 1)[1].strip().strip('"')
                elif "Search traceback:" in line:
                    in_search = True
                elif "Alignment traceback:" in line:
                    in_search = False
                elif in_search and "<TracebackItem(" in line:
                    match = TRACE.search(line)
                    if not match:
                        raise ValueError(f"Unparseable selected traceback: {line}")
                    lemma, am, lm, start, end = match.groups()
                    segment["items"].append(
                        dict(lemma=lemma, am=float(am), lm=float(lm), start=float(start), end=float(end))
                    )
    if segment is not None:
        yield segment


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return dict(count=0)

    def quantile(q):
        x = q * (len(values) - 1)
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)

    return dict(
        count=len(values),
        mean=statistics.mean(values),
        min=values[0],
        max=values[-1],
        **{f"p{p}": quantile(p / 100) for p in (5, 25, 50, 75, 95)},
    )


def read_lexicon(path):
    with uopen(path, "rt") as src:
        return ET.parse(src).getroot()


def lexical_piece_statistics(root):
    """Each lemma gets equal weight, then each of its pronunciations equal weight."""
    lemma_means = []
    lengths = []
    for lemma in root.findall("lemma"):
        if lemma.get("special") is not None or (lemma.find("synt") is not None and not lemma.findall("synt/tok")):
            continue
        ns = [len((phon.text or "").split()) for phon in lemma.findall("phon")]
        ns = [n for n in ns if n]
        if ns:
            lemma_means.append(statistics.mean(ns))
            lengths.extend(ns)
    if not lemma_means:
        raise ValueError("No ordinary lexical pronunciations")
    return dict(
        mean_pieces_per_word=statistics.mean(lemma_means),
        per_pronunciation=distribution(lengths),
        num_lemmata=len(lemma_means),
    )


def audit_config(path):
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(path.get_path())
    scorers = {s: dict(cfg[s]) for s in cfg.sections() if "label-scorer" in s}
    for section in cfg.sections():
        dotted = {k[len("label-scorer.") :]: v for k, v in cfg[section].items() if k.startswith("label-scorer.")}
        if dotted:
            scorers[section + ".label-scorer"] = dotted
    # This diagnostic has no way to recover hidden transition costs in a model.
    # Only accept the explicit acoustic-only no-op setup used by these prototypes.
    if not scorers or any(v.get("type") != "no-op" for v in scorers.values() if "type" in v):
        raise ValueError(f"Audit word-exit costs explicitly before using this setup: {scorers}")
    if any("transition" in k or "word-exit" in k for v in scorers.values() for k in v):
        raise ValueError("Nonzero/unspecified transition scorer requires a dedicated audit")
    # RASR may use dotted keys in the parent section.
    lm_scale = cfg.getfloat("lib-rasr.lm", "scale", fallback=0.0)
    search = dict(cfg["lib-rasr.search-algorithm"])
    return dict(lm_scale=lm_scale, word_exit_cost=0.0, label_scorers=scorers, search=search)


class OovScoringBalanceJob(Job):
    """Audit old baseline/collapsed runs and compute alpha_0 without rerunning ASR."""

    def __init__(self, lexicon, baseline_log, baseline_config, collapsed_log, collapsed_config):
        self.lexicon = lexicon
        self.baseline_log, self.baseline_config = baseline_log, baseline_config
        self.collapsed_log, self.collapsed_config = collapsed_log, collapsed_config
        self.out_report = self.output_path("scoring-balance.json")
        self.out_alpha0 = self.output_var("alpha0")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        root = read_lexicon(self.lexicon)
        stats = lexical_piece_statistics(root)
        ordinary = {
            orth.text
            for lemma in root.findall("lemma")
            if lemma.get("special") is None and (lemma.find("synt") is None or lemma.findall("synt/tok"))
            for orth in lemma.findall("orth")
        }
        reports = {}
        for name, log, config in [
            ("baseline", self.baseline_log, self.baseline_config),
            ("collapsed", self.collapsed_log, self.collapsed_config),
        ]:
            settings = audit_config(config)
            if name == "baseline" and any(
                float(settings["search"].get(key, "0")) != 0
                for key in ("unknown-word-penalty", "unknown-piece-penalty")
            ):
                raise ValueError("Calibrate alpha from a baseline without unknown penalties")
            costs, totals, words, reference_words = [], [], [], []
            for seg in read_tracebacks(log):
                previous = 0.0
                if not seg["items"]:
                    raise ValueError(f"Missing traceback for {seg['tag']}")
                for item in seg["items"]:
                    delta = item["lm"] - previous
                    previous = item["lm"]
                    if name == "baseline" and item["lemma"] in ordinary:
                        costs.append(delta)
                totals.append(seg["items"][-1]["lm"])
                words.append(len(seg["hypothesis"].split()))
                reference_words.append(len(seg["reference"].split()))
            reports[name] = dict(
                settings=settings,
                ordinary_word_lm_cost=distribution(costs),
                total_lm_cost_including_eos=distribution(totals),
                recognized_word_count=distribution(words),
                reference_word_count=distribution(reference_words),
            )
        if not reports["baseline"]["ordinary_word_lm_cost"]["count"]:
            raise ValueError("No ordinary word LM events in baseline")
        alpha0 = reports["baseline"]["ordinary_word_lm_cost"]["mean"] / stats["mean_pieces_per_word"]
        if not math.isfinite(alpha0) or alpha0 <= 0:
            raise ValueError(f"Unusable alpha_0={alpha0}; choose a working nonzero-LM baseline")
        report = dict(
            alpha0=alpha0,
            lexical_statistics=stats,
            runs=reports,
            convention="Scaled LM costs; baseline EOS excluded from per-word mean; lemma-weighted pieces",
        )
        with uopen(self.out_report, "wt") as dst:
            json.dump(report, dst, indent=2)
        self.out_alpha0.set(alpha0)


def ctc_viterbi_alignment(log_probs, labels, blank=0):
    """Exact CTC Viterbi forced alignment; return one state per acoustic frame.

    CPU numpy DP, O(T * number_of_reference_labels), no beam, no language model.
    Refuse impossible alignments instead of taking an argmax through -infinity.
    """
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels) or blank in labels:
        raise ValueError("Empty reference or reference containing CTC blank")
    states = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    states[1::2] = labels
    prev = np.full(len(states), -np.inf)
    prev[:2] = log_probs[0, states[:2]]
    back = np.zeros((len(log_probs), len(states)), dtype=np.int8)
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for t in range(1, len(log_probs)):
        steps = np.stack([prev, np.r_[-np.inf, prev[:-1]], np.r_[-np.inf, -np.inf, prev[:-2]]])
        steps[2, ~skip] = -np.inf
        back[t] = np.argmax(steps, axis=0)
        prev = np.max(steps, axis=0) + log_probs[t, states]
    state = len(states) - 2 + int(prev[-1] > prev[-2])
    if not np.isfinite(prev[state]):
        raise ValueError("Reference has no finite CTC alignment")
    path = np.zeros(len(log_probs), dtype=np.int64)
    for t in range(len(log_probs) - 1, -1, -1):
        path[t] = state
        state -= back[t, state]
    return path


class GraniteWordStartMarginJob(Job):
    """Forced-reference boundary margins on the *same* dev-other dataset/frontend.

    Uses lower-case reference text, explicitly recorded in the report, not a
    model-generated transcript. Samples evenly across the sorted sequence order.
    Unmarked alternatives must exist as exact vocabulary entries. Their absence
    is reported, not converted to the tokenizer's unknown/blank ID.
    """

    def __init__(self, repo, data_config, returnn_root, correctness_report, num_sequences=32):
        if num_sequences < 1:
            raise ValueError(num_sequences)
        self.repo, self.data_config, self.returnn_root = repo, data_config, returnn_root
        self.correctness_report = correctness_report
        self.num_sequences = num_sequences
        self.out_report = self.output_path("word-start-margins.json")
        self.out_samples = self.output_path("word-start-samples.jsonl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "gpu": 1, "gpu_mem": 11, "time": 2})

    def run(self):
        import sys
        import numpy as np
        import torch
        from i6_core.util import instanciate_delayed
        from .huggingface import _load_tokenizer
        from ..model_pipelines.granite_ctc.pytorch_modules import GraniteCtcRecogConfig, GraniteCtcRecogModel

        with uopen(self.correctness_report, "rt") as src:
            if not json.load(src).get("passed"):
                raise ValueError("Pinned decoder correctness checks have not passed")
        sys.path.insert(0, self.returnn_root.get_path())
        from returnn.datasets import init_dataset

        cfg = self.data_config.get_returnn_data("forward_data")
        dataset = init_dataset(instanciate_delayed(cfg.config["forward_data"]))
        dataset.init_seq_order(epoch=1)
        indices = np.linspace(0, dataset.num_seqs - 1, min(self.num_sequences, dataset.num_seqs), dtype=int).tolist()
        tokenizer = _load_tokenizer(self.repo)
        vocab = tokenizer.get_vocab()
        model = GraniteCtcRecogModel(GraniteCtcRecogConfig(huggingface_repo_dir=self.repo)).cuda().eval()
        margins, span_margins, counts, tags = [], [], Counter(), []
        with torch.inference_mode(), uopen(self.out_samples, "wt") as dst:
            for sample_index, i in enumerate(indices):
                dataset.load_seqs(i, i + 1)
                tag = dataset.get_tag(i)
                tags.append(tag)
                raw = dataset.get_data(i, "raw")
                reference = raw.item() if hasattr(raw, "item") else raw
                if isinstance(reference, bytes):
                    reference = reference.decode("utf-8")
                text = str(reference).lower()
                labels = tokenizer.encode(text, add_special_tokens=False)
                pieces = tokenizer.convert_ids_to_tokens(labels)
                audio = torch.as_tensor(dataset.get_data(i, "data"), dtype=torch.float32, device="cuda")
                if audio.ndim == 1:
                    audio = audio[:, None]
                costs = model(audio[None], torch.tensor([len(audio)], device="cuda"))[0].cpu().numpy()
                alignment = ctc_viterbi_alignment(-costs, labels)
                for j, piece in enumerate(pieces):
                    if not piece.startswith("▁") or piece == "▁":
                        continue
                    counts["word_start_pieces"] += 1
                    unmarked = vocab.get(piece[1:])
                    if unmarked is None or unmarked == 0 or unmarked in tokenizer.all_special_ids:
                        counts["no_unmarked_counterpart"] += 1
                        continue
                    frames = np.flatnonzero(alignment == 2 * j + 1)
                    if not len(frames):
                        raise ValueError(f"Unvisited reference label in {tag}")
                    # Positive margin means replacing the marked token costs more.
                    deltas = costs[frames, unmarked] - costs[frames, labels[j]]
                    peak = int(np.argmin(costs[frames, labels[j]]))
                    margin = float(deltas[peak])
                    margins.append(margin)
                    span_margins.append(float(deltas.sum()))
                    dst.write(
                        json.dumps(
                            dict(
                                tag=tag,
                                reference=text,
                                piece=piece,
                                marked_id=labels[j],
                                unmarked_id=unmarked,
                                frames=frames.tolist(),
                                peak_margin=margin,
                                aligned_span_margin=float(deltas.sum()),
                            )
                        )
                        + "\n"
                    )
                print(f"Aligned boundary diagnostic {sample_index + 1}/{len(indices)}: {tag}", flush=True)
        if not margins:
            raise ValueError("No paired marked/unmarked boundary samples")
        with uopen(self.out_report, "wt") as dst:
            json.dump(
                dict(
                    tags=tags,
                    sequence_indices=indices,
                    counts=dict(counts),
                    peak_frame_margin=distribution(margins),
                    aligned_span_margin=distribution(span_margins),
                    reference_policy="lowercase, no HF scoring normalizer before tokenization",
                    margin="cost(unmarked)-cost(marked); positive favors a boundary",
                    dtype="float32",
                    preemphasis=self.data_config.preemphasis,
                    peak_normalization=self.data_config.peak_normalization,
                ),
                dst,
                indent=2,
            )
