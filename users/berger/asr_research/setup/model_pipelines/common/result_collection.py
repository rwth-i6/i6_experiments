"""Live reports over exported results, including experiments no longer scheduled.

No recognition jobs are reconstructed. Missing exports remain visible and do
not prevent reporting completed experiments. Read small summaries only, never
hypotheses, tracebacks or recognition logs.
"""

import csv
import io
import json
import re
import time
from pathlib import Path

from sisyphus import gs, tk


GROUPS = ("granite-ctc", "parakeet-rnnt", "cohere-aed", "qwen-granite", "qwen-parakeet-rnnt", "qwen-speech-0.6b",
          "qwen-granite-1.7b", "qwen-parakeet-rnnt-1.7b", "qwen-speech-1.7b",
          "qwen-parakeet-rnnt-speech-1.7b")
_REGISTERED = {}

MODELS = {
    "granite-ctc": "Granite CTC",
    "parakeet-rnnt": "Parakeet RNN-T",
    "cohere-aed": "Cohere AED",
}
TEXT_QWEN = "Text Qwen 0.6B"
SPEECH_QWEN = "Speech Qwen 0.6B"


def _corpus(dataset):
    if dataset.startswith("librispeech-"):
        return "librispeech"
    if dataset.startswith("loquacious-"):
        return "loquacious"
    return "open_asr"


def describe(row):
    """Explicit labels from audited experiment definitions, not suite names.

    Unknown historical conditions retain their id with an unverified-method
    label rather than guessing what ran. Original ids/provenance stay available.
    """
    group, condition = row["group"], row["condition"]
    speech_qwen = "Speech Qwen 1.7B" if group in {"qwen-speech-1.7b", "qwen-parakeet-rnnt-speech-1.7b"} else SPEECH_QWEN
    if group == "qwen-parakeet-rnnt-speech-1.7b":
        group = "qwen-parakeet-rnnt"
    elif group == "qwen-speech-1.7b":
        group = "qwen-speech-0.6b"
    text_qwen = "Text Qwen 1.7B" if group.endswith("-1.7b") else TEXT_QWEN
    group = group.removesuffix("-1.7b")
    family = {"qwen-granite": "granite-ctc", "qwen-parakeet-rnnt": "parakeet-rnnt"}.get(group, group)
    model = MODELS.get(family, group)
    lm, method, lexicon = "None", "Unverified", "—"
    if group.startswith("own-"):
        model = row["model"]
        method, lexicon = "1-pass fusion", "None"
        if "ls-word-trafo" in condition:
            lm = "LS word Trafo"
        elif "ls-bpe-trafo" in condition:
            lm = "LS BPE Trafo"
        elif "qwen-" in condition:
            size = re.search(r"qwen-([\d.]+b)", condition)
            lm = "Text Qwen " + (size[1].upper() if size else "unknown size")
        else:
            method = "RASR greedy" if condition == "greedy" else "RASR lex-free"
        return model, lm, method, lexicon
    speech_native = condition == "speech-native" or (group == "qwen-speech-0.6b" and condition.startswith("native"))
    if speech_native:
        return speech_qwen, "None", "Native beam" if "-beam" in condition else "Native", "None"
    if condition.startswith(("first-pass-speech-nbest", "two-pass-speech", "oracle-speech")):
        if group == "qwen-speech-0.6b":
            model = MODELS["granite-ctc"]
        lexicon = "LS (open)" if model == MODELS["granite-ctc"] else "None"
        if condition.startswith("first-pass-"):
            return model, "None", "1st-pass control", lexicon
        n = re.search(r"-n(\d+)", condition)[1]
        if condition.startswith("oracle-"):
            return model, "Oracle", f"Oracle ({n}-best)", lexicon
        return model, speech_qwen, f"2-pass ({n}-best)", lexicon
    if group == "qwen-speech-0.6b":
        model, lm = MODELS["granite-ctc"], speech_qwen
        method, lexicon = "1-pass fusion", "None"
    elif condition == "native":
        method, lexicon = "Native", "None"
    elif condition == "rasr-lexfree" or condition.startswith("zero-"):
        method = "RASR greedy" if condition == "zero-greedy" else "RASR lex-free"
        lexicon = "None"
    elif group == "qwen-parakeet-rnnt":
        lm = speech_qwen if condition.startswith("speech-qwen") else text_qwen
        method, lexicon = "1-pass fusion", "None"
        if "earnings-prune" in condition:
            method = "1-pass (Earnings tuning)"
    elif group == "qwen-granite" and condition.startswith("one-pass"):
        lm, method, lexicon = text_qwen, "1-pass fusion", "None"
    elif group == "qwen-granite" and condition.startswith(("first-pass-", "two-pass-")):
        lexicon = "Corpus (open)" if "corpus-lexicon" in condition else "LS (open)"
        method = "1st-pass control"
        if condition.startswith("two-pass-"):
            lm, method = text_qwen, "2-pass (50-best)"
    elif "tree" in condition:
        lexicon = "Corpus" if "corpus" in condition else "LS"
        opened = "tree-open" in condition and "tree-open-control" not in condition
        method = "Open tree" if opened else "Closed tree"
        if "tree-open-control" in condition:
            method = "Closed-tree control"
        if "trafo" in condition:
            lm = "LS word Trafo"
        elif "corpus4gram" in condition or condition.startswith("corpus-tree-4gram"):
            lm = "Corpus 4-gram"
        elif "4gram" in condition:
            lm = "LS 4-gram"
        elif "qwen" in condition:
            lm = TEXT_QWEN
    return model, lm, method, lexicon


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), "available"
    except FileNotFoundError:
        return {}, "no completed export"
    except (OSError, ValueError) as exc:
        return {}, f"unreadable: {type(exc).__name__}"


def _number(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return (
        "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            + ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
        )
        + "\n"
    )


class ResultCollection:
    def __init__(self, output_root):
        self.root = Path(output_root)
        self._loaded_at = None
        self._rows = []

    def rows(self):
        # All registered reports share one scan during an update cycle.
        now = time.monotonic()
        if self._loaded_at is not None and now - self._loaded_at < 60:
            return self._rows
        rows = []
        for group in GROUPS:
            base = self.root / "open-asr" / group
            if not base.is_dir():
                continue
            for condition in sorted(base.iterdir()):
                if not condition.is_dir():
                    continue
                for dataset in sorted(condition.iterdir()):
                    if not dataset.is_dir():
                        continue
                    score_path = dataset / "scoring"
                    # A dangling scoring link is an unfinished exported result.
                    if not score_path.exists() and not score_path.is_symlink():
                        continue
                    score, status = _json(score_path)
                    runtime, _ = _json(dataset / "runtime")
                    if status == "available" and not isinstance(score.get("wer_percent"), (int, float)):
                        status = "invalid scoring summary"
                    excluded = "corpus-voxpopuli-cleaned-ls" in condition.name
                    if group == "cohere-aed" and "tree" in condition.name:
                        excluded = True  # Backlogged label-sync tree implementation.
                    rows.append(
                        dict(
                            group=group,
                            condition=condition.name,
                            dataset=dataset.name,
                            status=status,
                            excluded=excluded,
                            wer=score.get("wer_percent"),
                            ins=score.get("ins"),
                            delete=score.get("del"),
                            sub=score.get("sub"),
                            ref_len=score.get("ref_len"),
                            rtf=runtime.get("rtf"),
                            normalizer=score.get("normalizer_revision"),
                            source=str(score_path),
                            score_job=str(score_path.resolve(strict=False).parent.parent),
                        )
                    )
        # Own-model results are read from completed score exports directly:
        # the aggregate Sisyphus table may not yet have refreshed.
        for context in ("librispeech", "loquacious_small", "loquacious_medium"):
            base = self.root / context / "recognition"
            for experiment in sorted(base.glob("*__recog_wordwise-*")):
                name, condition = experiment.name.split("__recog_wordwise-", 1)
                for dataset in sorted(experiment.iterdir()):
                    report = dataset / "scoring_reports"
                    if not report.exists() and not report.is_symlink():
                        continue
                    output = report.resolve(strict=False).parent
                    score, status = _json(output / "wer")
                    if status == "available" and not isinstance(score, (int, float)):
                        status = "invalid scoring summary"
                    def value(filename):
                        result, _ = _json(output / filename)
                        return result if isinstance(result, (int, float)) else None
                    rtf, _ = _json(dataset / "total_rtf")
                    family = {
                        "ctc_bpe": "BPE CTC",
                        "ffnn_transducer_bpe": "Context-1 mRNN-T",
                        "ffnn_transducer_pruned_bpe": "Context-1 mRNN-T (pruned)",
                        "full_ctx_transducer_bpe": "Full-context mRNN-T",
                    }.get(name, name)
                    corpus = "librispeech" if context == "librispeech" else "loquacious"
                    training_corpus = {
                        "librispeech": "LibriSpeech",
                        "loquacious_small": "Loquacious-small",
                        "loquacious_medium": "Loquacious-medium",
                    }[context]
                    rows.append(dict(
                        group=f"own-{context}", condition=condition, dataset=f"{corpus}-{dataset.name}",
                        model=f"{training_corpus} {family}",
                        status=status, excluded=False, wer=score if isinstance(score, (int, float)) else None,
                        ins=value("num_insertions"), delete=value("num_deletions"), sub=value("num_substitution"),
                        ref_len=value("ref_words"), rtf=rtf if isinstance(rtf, (int, float)) else None,
                        normalizer=None, source=str(report), score_job=str(output.parent),
                    ))
        for row in rows:
            row["model"], row["lm"], row["decoding"], row["lexicon"] = describe(row)
            row["corpus"] = _corpus(row["dataset"])
        self._rows, self._loaded_at = rows, now
        return rows

    def index(self):
        rows = self.rows()
        completed = sum(r["status"] == "available" for r in rows)
        return (
            "# ASR / LM experiment reports\n\n"
            "Live collection of existing exported results, including inactive historical conditions. "
            "No experiments are scheduled by this collection. Refresh interval: five minutes.\n\n"
            f"Scoring exports found: {len(rows)}; readable completed summaries: {completed}.\n\n"
            "- [ASR, text-Qwen and Speech-Qwen comparisons](asr-combinations.md)\n"
            "- [LibriSpeech: standalone and combination results](librispeech.md)\n"
            "- [Loquacious: full splits and source sub-splits](loquacious.md)\n"
            "- [Open ASR: per-corpus standalone and combination results](open_asr.md)\n"
            "- [Model IDs and condition terminology](models-and-methods.md)\n"
            "- [Lexicon and specialized word-LM comparisons](word-lms.md)\n"
            "- [Own BPE CTC / monotonic transducer pilots](own-models.md)\n"
            "- [Complete inventory, including incomplete and excluded conditions](inventory.md)\n"
            "- [Machine-readable results and provenance](all-results.tsv)\n\n"
            "Controlled research-plan comparisons are registered separately by `config/controlled_comparison.py`: "
            "[development-selected audit](controlled-comparison.md), followed by "
            "[exact selected evaluations](controlled-selected-evaluation.md) after the second-stage config runs.\n\n"
            "Research-plan Step 2 adds [error complementarity](error-complementarity.md) and "
            "[selected-evaluation attribution](error-complementarity-selected.md), using CPU analysis only.\n\n"
            "WER is percent; lower is better. ΔWER is fusion minus the best displayed standalone "
            "native/beam control for that ASR; Speech-Qwen combinations also compare against "
            "standalone Speech-Qwen and use the better baseline. This is a comparison to "
            "observed controls, not selection of a fusion setting on test. Differences are "
            "in percentage points, not relative percent. It is shown only for matching, explicitly "
            "recorded normalizer revisions. Different search configurations, casing policies and "
            "hardware are not controlled by this report; small differences are not significance tests. "
            "Speech-Qwen fusion is audio-conditioned model combination, not a text-only LM. "
            "Corpus subsets overlap parent datasets; no pooled cross-corpus WER is reported.\n\n"
            "A missing score is not automatically a failed job. Only registered scoring exports "
            "can be inventoried; unregistered experiments are not inferred. Invalid VoxPopuli/"
            "LibriSpeech word-LM pairings and backlogged AED tree runs appear only in the inventory.\n"
        )

    def comparisons(self, *, word_lms=False):
        return "\n".join(self.corpus_report(c, word_lms=word_lms) for c in ("librispeech", "loquacious", "open_asr"))

    def corpus_report(self, corpus, *, word_lms=False, own_only=False):
        title = {"librispeech": "LibriSpeech", "loquacious": "Loquacious", "open_asr": "Open ASR suite"}[corpus]
        rows = [r for r in self.rows() if r["corpus"] == corpus and not r["excluded"]]
        if own_only:
            rows = [r for r in rows if r["group"].startswith("own-")]
        if word_lms:
            rows = [r for r in rows if r["group"] in GROUPS[:3]]
        lines = [f"# {title}: standalone ASR and model combinations\n",
                 "WER %. [Labels](models-and-methods.md) · [Notes](index.md). "
                 "All settings shown; subsets overlap.\n"]
        for dataset in sorted({r["dataset"] for r in rows}):
            lines.append(f"## {dataset}\n")
            selected, seen = [], set()
            for row in rows:
                if row["dataset"] != dataset:
                    continue
                # Native ASR / Speech-ASR scores are exported by several suites.
                # Deduplicate identical results only; preserve all differing runs.
                key = (row["model"], row["lm"], row["decoding"], row["lexicon"], row["wer"],
                       row["normalizer"], row["ref_len"], row["status"])
                if row["condition"] in {"native", "speech-native"} and key in seen:
                    continue
                seen.add(key)
                selected.append(row)
            for label, combined in (("Standalone / no-LM", False), ("Combinations", True)):
                subset = [r for r in selected if (r["lm"] != "None") == combined]
                if not subset:
                    continue
                lines.append(f"### {label}\n")
                rendered = []
                for r in sorted(subset, key=lambda r: (r["model"], r["lm"], r["condition"])):
                    baseline_models = {r["model"]}
                    if r["lm"].startswith("Speech Qwen "):
                        baseline_models.add(r["lm"])
                    baseline_rows = [b for b in selected if b["model"] in baseline_models
                                     and b["lm"] == "None"
                                     and (b["condition"].startswith("native") or b["condition"] == "speech-native"
                                          or b["condition"].startswith("zero-"))
                                     and r["normalizer"] and b["normalizer"] == r["normalizer"]
                                     and isinstance(b["wer"], (int, float))]
                    baseline = min(baseline_rows, key=lambda b: b["wer"], default=None)
                    delta = r["wer"] - baseline["wer"] if baseline and isinstance(r["wer"], (int, float)) else None
                    settings = []
                    for pattern, name in ((r"(?:lm|scale-)([\d.]+)", "LM"),
                                          (r"-thr-?([\d.]+)", "thr"),
                                          (r"-beam(\d+)", "beam"), (r"-pre(\d+)", "pre"),
                                          (r"-prior([\d.]+)", "prior"), (r"-ilm([\d.]+)", "ILM"),
                                          (r"-lp([\d.]+)", "length penalty"),
                                          (r"-word-penalty(-?[\d.]+)", "word penalty"),
                                          (r"-unk-beta([\d.]+)", "UNK")):
                        match = re.search(pattern, r["condition"])
                        if match:
                            settings.append(f"{name} {match[1]}")
                    if r["condition"] == "speech-qwen-fast":
                        settings.extend(["LM 0.3", "beam 8", "pre 32", "thr 3"])
                    source = f"[result]({r['source']})"
                    rendered.append((r["model"], r["lm"], r["decoding"], r["lexicon"],
                                     _number(r["wer"]), _number(delta), _number(r["rtf"], 3),
                                     "; ".join(settings) or "Default", r["status"], source))
                lines.append(_table(["ASR", "LM / model", "Search", "Lexicon",
                                     "WER %", "ΔWER pp", "RTF", "Settings", "Status", "Source"], rendered))
        if not rows:
            lines.append("No scoring exports found for this corpus yet.\n")
        return "\n".join(lines)

    def terminology(self):
        return (
            "# Models and decoding terminology\n\n"
            "## Short table labels\n\n"
            "LS = LibriSpeech; Trafo = Transformer (96 layers for these LMs); "
            "mRNN-T = monotonic transducer. The corpus prefix of an own-model name "
            "is its training corpus; Context-1 denotes one-label prediction context.\n\n"
            "Native = standalone native decoding; RASR lex-free = lexicon-free RASR search; "
            "1-pass fusion = dynamic wordwise shallow fusion; 2-pass (50-best) = "
            "n-best generation followed by Qwen rescoring; 1st-pass control = no rescoring. "
            "Closed-tree control uses the open-tree lexicon with fallback disabled. "
            "Open tree enables fallback. Text/Speech Qwen labels distinguish text-only "
            "from audio-conditioned scoring, with parameter size always included. "
            "UNK settings are additive costs in nats. Default denotes an unsuffixed "
            "export setting, not necessarily identical pruning across search methods.\n\n"
            "Prior = CTC posterior-prior subtraction; ILM = zero-encoder RNN-T internal-LM "
            "subtraction (conditional on nonblank; blanks unchanged). Correction grids are "
            "dev-only. Native beam 4/8 Speech-Qwen controls use length penalty 1; lp0 denotes "
            "the dev-only unnormalized-score ablation.\n\n"
            "Speech-Qwen n-best rescoring teacher-forces the entire candidate with its audio prompt "
            "and one turn-end token. `original` preserves ASR spelling; `initial-capital-period` "
            "is a sentence-formatting ablation, not scoring normalization. First-pass costs are retained, "
            "including Granite open-tree fallback costs. Parakeet candidates use Viterbi recombination, "
            "not alignment-sum decoding. Two-pass RTF includes all cached scoring variants and an "
            "independent one-best parity decode. Oracle rows use references and are not deployable systems. "
            "Prune-threshold rows are dev budget diagnostics; beam tightening follows threshold selection.\n\n"
            "## Exact pretrained model IDs\n\n"
            "- Granite: `ibm-granite/granite-speech-5.0-470m-turboctc-nc`.\n"
            "- Parakeet: `nvidia/parakeet-rnnt-1.1b`.\n"
            "- Cohere AED: `CohereLabs/cohere-transcribe-03-2026`.\n"
            "- Text Qwen in the open-ASR fusion suites: `Qwen/Qwen3-0.6B-Base`.\n"
            "- Speech Qwen: `Qwen/Qwen3-ASR-0.6B-hf` and `Qwen/Qwen3-ASR-1.7B-hf`; both see the audio.\n"
            "- Development pilots also use text `Qwen/Qwen3-1.7B-Base` with Granite, Parakeet and own models; "
            "the size is explicit in each table row. Own CTC and monotonic transducer "
            "training corpora are explicit too.\n\n"
            "## Original export names\n\n"
            "`qwen-granite` is an export directory, not a model name: it contains Granite "
            "standalone controls and combinations with text Qwen3-0.6B-Base.\n\n"
            "`native` means standalone decoding with the model's native implementation. "
            "The model depends on the directory: Granite / Parakeet / Cohere in their suites, "
            "and Qwen3-ASR of the indicated size in `qwen-speech-0.6b` / `qwen-speech-1.7b`. "
            "It never means text-Qwen standalone ASR.\n\n"
            "`speech-native` is another export of standalone Qwen3-ASR-0.6B, not Parakeet fusion. "
            "Identical duplicated standalone results are displayed once per dataset.\n\n"
            "`one-pass` / `text-qwen`: dynamic wordwise text-LLM shallow fusion without a lexicon. "
            "`speech-qwen` / `granite-fusion`: the same type of combination with an "
            "audio-conditioned Speech-Qwen model.\n\n"
            "`two-pass-librispeech-lexicon`: Granite generates up to 50 n-best candidates "
            "using the LibriSpeech open tree, without a word LM; text Qwen3-0.6B-Base then "
            "rescores the candidates. `two-pass-corpus-lexicon` uses a corpus-specific open "
            "tree instead. The lexicon name does not mean a LibriSpeech LM is used. "
            "`first-pass-…` is the corresponding unrescored ASR control.\n\n"
            "`tree-open-control` disables fallback; `tree-open` enables OOV fallback. "
            "A LibriSpeech LM is domain-matched on LibriSpeech and transferred elsewhere; "
            "corpus-specific 4-grams are labelled separately. Exact conditions, scoring "
            "normalizer revisions and job paths remain in [the inventory](inventory.md) "
            "and [TSV](all-results.tsv). Missing scores do not by themselves mean job failure.\n"
        )

    def _legacy_comparisons(self, *, word_lms=False):
        rows = self.rows()
        baselines = {(r["group"], r["dataset"]): r for r in rows if r["condition"] == "native"}
        # Loquacious native exports live in the fusion suites rather than the
        # original eight-corpus benchmark directories.
        for suite, model in (("qwen-granite", "granite-ctc"), ("qwen-parakeet-rnnt", "parakeet-rnnt")):
            for r in rows:
                if r["group"] == suite and r["condition"] == "native":
                    baselines.setdefault((model, r["dataset"]), r)
        lines = [
            "# " + ("Lexicon / specialized word-LM results" if word_lms else "ASR / Qwen combinations"),
            "",
            "See [report notes](index.md). All available conditions are retained; no best-on-test selection.\n",
        ]
        if word_lms:
            lines.append(
                "LibriSpeech word LMs are domain-matched on LibriSpeech, transferred elsewhere. "
                "`corpus4gram` / `corpus-tree-4gram` denote corpus-specific LMs. "
                "Open-tree no-LM controls can still carry fallback penalties.\n"
            )
        groups = GROUPS[:3] if word_lms else GROUPS[3:]
        for group in groups:
            selected = [
                r
                for r in rows
                if r["group"] == group
                and not r["excluded"]
                and ((r["condition"] in {"native", "rasr-lexfree"} or "tree" in r["condition"]) if word_lms else True)
            ]
            if not selected:
                continue
            rendered = []
            for r in sorted(selected, key=lambda r: (r["dataset"], r["condition"])):
                baseline_group = (
                    group
                    if word_lms
                    else {
                        "qwen-granite": "granite-ctc",
                        "qwen-parakeet-rnnt": "parakeet-rnnt",
                        "qwen-speech-0.6b": "granite-ctc" if r["condition"].startswith("granite-fusion") else group,
                    }[group]
                )
                baseline = baselines.get((baseline_group, r["dataset"]))
                delta = None
                if baseline and r["normalizer"] and baseline["normalizer"] == r["normalizer"]:
                    if isinstance(r["wer"], (int, float)) and isinstance(baseline["wer"], (int, float)):
                        delta = r["wer"] - baseline["wer"]
                rendered.append(
                    (r["dataset"], r["condition"], _number(r["wer"]), _number(delta), _number(r["rtf"], 3), r["status"])
                )
            lines.extend(
                [f"## {group}\n", _table(["Dataset", "Condition", "WER %", "ΔWER pp", "RTF", "Status"], rendered)]
            )
        return "\n".join(lines) + "\n"

    def own_models(self):
        if any(r["group"].startswith("own-") for r in self.rows()):
            return (
                "# Own trained models: development results\n\n"
                "Read directly from individual scoring exports, not the potentially stale aggregate table. "
                "WER exports are rounded. Normalizer revisions are not exported for these jobs, "
                "so no automatic ΔWER is shown. Historical non-wordwise experiments remain in "
                "the original corpus report.txt files.\n\n"
                + self.corpus_report("librispeech", own_only=True)
                + self.corpus_report("loquacious", own_only=True)
                + "\n## Historical corpus reports\n\n"
                + "\n".join(
                    f"- [{corpus}]({self.root / corpus / 'report.txt'})"
                    for corpus in ("librispeech", "loquacious_small", "loquacious_medium")
                ) + "\n"
            )
        lines = [
            "# Own-model development pilots\n",
            "Source: `wordwise-lm/dev/summary`. "
            "These existing reports use rounded WER; unavailable entries remain blank. "
            "Development results are not held-out benchmark estimates.\n",
        ]
        path = self.root / "wordwise-lm/dev/summary"
        try:
            table = path.read_text(encoding="utf-8")
        except OSError:
            table = "No own-model summary is available yet."
        rendered = []
        for line in table.splitlines()[2:]:
            fields = [field.strip() for field in line.split("|")]
            if len(fields) >= 9:
                rendered.append(
                    (
                        fields[0].split("__", 1)[-1],
                        fields[1],
                        fields[2] or "—",
                        fields[3] or "—",
                        fields[4] or "—",
                        fields[5] or "—",
                        fields[8] or "—",
                    )
                )
        lines.append(
            _table(["Experiment", "Dataset", "WER %", "Del %", "Ins %", "Sub %", "Total RTF"], rendered)
            if rendered
            else table
        )
        lines.append("## Historical corpus reports\n")
        for corpus in ("librispeech", "loquacious_small", "loquacious_medium"):
            source = self.root / corpus / "report.txt"
            lines.append(f"- [{corpus}]({source})\n")
        return "\n".join(lines)

    def inventory(self):
        return "# Export inventory\n\n" + _table(
            ["Group", "Condition", "Dataset", "WER %", "Status", "Excluded", "Scoring source"],
            [
                (
                    r["group"],
                    r["condition"],
                    r["dataset"],
                    _number(r["wer"]),
                    r["status"],
                    "yes" if r["excluded"] else "no",
                    r["source"],
                )
                for r in self.rows()
            ],
        )

    def tsv(self):
        out = io.StringIO()
        fields = [
            "corpus", "model", "lm", "decoding", "lexicon",
            "group",
            "condition",
            "dataset",
            "status",
            "excluded",
            "wer",
            "ins",
            "delete",
            "sub",
            "ref_len",
            "rtf",
            "normalizer",
            "source",
            "score_job",
        ]
        writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(self.rows())
        return out.getvalue()


def register_result_collection(*, output_root=None, report_prefix="reports/asr-lm"):
    """Register periodic reports only; do not add recognition-job dependencies."""
    root = Path(output_root or gs.OUTPUT_DIR).absolute()
    key = (id(tk.sis_graph), str(root), report_prefix)
    if key in _REGISTERED:
        return _REGISTERED[key]
    collection = ResultCollection(root)
    for name, render in (
        ("index.md", collection.index),
        ("asr-combinations.md", collection.comparisons),
        ("word-lms.md", lambda: collection.comparisons(word_lms=True)),
        ("librispeech.md", lambda: collection.corpus_report("librispeech")),
        ("loquacious.md", lambda: collection.corpus_report("loquacious")),
        ("open_asr.md", lambda: collection.corpus_report("open_asr")),
        ("models-and-methods.md", collection.terminology),
        ("own-models.md", collection.own_models),
        ("inventory.md", collection.inventory),
        ("all-results.tsv", collection.tsv),
    ):
        tk.register_report(f"{report_prefix}/{name}", render, required=[], update_frequency=300)
    _REGISTERED[key] = collection
    return collection
