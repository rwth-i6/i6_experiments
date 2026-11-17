"""Descriptive surface-form audit of frozen hypotheses; no forced scoring or rewrites."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from sisyphus import Job, Task

from .hf_scoring import normalize_hf_english


NUMBER_WORDS = set("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                   "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                   "hundred thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth".split())
PATTERNS = {
    "contraction": r"\b\w+(?:n't|'(?:s|d|re|ve|ll|m))\b",
    "ambiguous_s": r"\b\w+'s\b", "ambiguous_d": r"\b\w+'d\b",
    "hyphen": r"\b\w+(?:-\w+)+\b", "dotted_abbreviation": r"\b(?:[A-Za-z]\.){2,}",
    "title": r"\b(?:Dr|Mr|Mrs|Ms)\.(?=\s|$)", "digit": r"\b\w*\d\w*\b",
    "apostrophe": r"['’]", "sentence_punctuation": r"[,.!?;:]",
}


def surface_features(text):
    canonical = text.replace("’", "'")
    counts = {key: len(re.findall(pattern, canonical, flags=re.IGNORECASE)) for key, pattern in PATTERNS.items()}
    words = re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", text)
    counts.update(words=len(words), uppercase_words=sum(any(c.isupper() for c in w) for w in words),
                  allcaps_words=sum(len(w) > 1 and w.isalpha() and w.isupper() for w in words),
                  number_word=sum(w.lower() in NUMBER_WORDS for w in words))
    return counts


def jsonl_map(path, fields=None):
    result = {}
    with open(path) as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in result:
                raise ValueError(f"Duplicate ID in {path}")
            result[row["id"]] = {k: row[k] for k in fields} if fields else row
    return result


def grouped_raw(record):
    hyps, refs = jsonl_map(record["hypotheses"], fields=('id', 'text')), jsonl_map(record["references"])
    if hyps.keys() != refs.keys():
        raise ValueError("Raw hypothesis/reference coverage differs")
    groups = {}
    for seq_id, ref in refs.items():
        groups.setdefault(ref["group_id"], []).append((ref["chunk_index"], hyps[seq_id]["text"]))
    return {group: " ".join(text for _, text in sorted(parts)) for group, parts in groups.items()}


def paired_feature_rows(am, speech, fusion, raw_speech):
    """Slices are defined from S's output: descriptive, not independent truth labels."""
    if not (am.keys() == speech.keys() == fusion.keys() == raw_speech.keys()):
        raise ValueError("Paired scoring-unit coverage differs")
    features = {seq_id: surface_features(text) for seq_id, text in raw_speech.items()}
    selectors = {"all": lambda f: True}
    for key in ("contraction", "ambiguous_s", "ambiguous_d", "hyphen", "dotted_abbreviation", "title", "allcaps_words", "digit", "number_word"):
        selectors[key + "/yes"] = lambda f, k=key: f[k] > 0
        selectors[key + "/no"] = lambda f, k=key: f[k] == 0
    rows = []
    for label, predicate in selectors.items():
        ids = [seq_id for seq_id, f in features.items() if predicate(f)]
        if not ids:
            continue
        totals = {name: {k: sum(units[i]["counts"][k] for i in ids) for k in ("total", "ref_len", "ins", "del", "sub")}
                  for name, units in (("am", am), ("speech", speech), ("fusion", fusion))}
        wer = {k: 100 * v["total"] / v["ref_len"] for k, v in totals.items() if v["ref_len"]}
        rows.append(dict(slice=label, units=len(ids), counts=totals, wer=wer,
                         delta_vs_better=wer["fusion"] - min(wer["am"], wer["speech"]),
                         improved_vs_s=sum(fusion[i]["counts"]["total"] < speech[i]["counts"]["total"] for i in ids),
                         worsened_vs_s=sum(fusion[i]["counts"]["total"] > speech[i]["counts"]["total"] for i in ids)))
    return rows


class ConventionAuditJob(Job):
    def __init__(self, records, *, protocol_sha256, code_version):
        self.records, self.protocol_sha256, self.code_version = records, protocol_sha256, code_version
        self.out_summary = self.output_path("summary.json")
        self.out_report = self.output_path("report.md")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=2))

    def run(self):
        from .controlled_comparison import NORMALIZER
        systems, panels, sources = [], {}, []
        for r in self.records:
            if r["normalizer"] != NORMALIZER:
                raise ValueError("Unexpected scoring normalizer")
            path = lambda k: r[k].get_path() if hasattr(r[k], "get_path") else r[k]
            raw = grouped_raw({k: path(k) for k in ("hypotheses", "references")})
            scored = jsonl_map(path("segments"))
            if raw.keys() != scored.keys():
                raise ValueError("Raw/scored group coverage differs")
            totals, present = Counter(), Counter()
            for text in raw.values():
                f = surface_features(text)
                totals.update(f)
                present.update(k for k, v in f.items() if v)
            systems.append(dict(dataset=r["dataset"], family=r["family"], units=len(raw),
                                occurrences=dict(totals), units_with_feature=dict(present),
                                raw_differs_from_hf=sum(text != normalize_hf_english(text) for text in raw.values())))
            panels.setdefault(r["dataset"], {})[r["family"]] = (scored, raw)
            sources.append(dict(dataset=r["dataset"], family=r["family"],
                                files={k: dict(path=path(k), sha256=hashlib.sha256(Path(path(k)).read_bytes()).hexdigest())
                                       for k in ("hypotheses", "references", "segments")}))
        paired = []
        for dataset, records in sorted(panels.items()):
            speech_key = "Speech Qwen 1.7B | None | standalone"
            for model in ("Granite CTC", "Parakeet RNN-T"):
                am_key, fused_key = model + " | None | standalone", model + " | Speech Qwen 1.7B | one-pass"
                if not {speech_key, am_key, fused_key} <= records.keys():
                    continue
                paired.append(dict(dataset=dataset, model=model,
                                   slices=paired_feature_rows(records[am_key][0], records[speech_key][0],
                                                             records[fused_key][0], records[speech_key][1])))
        result = dict(protocol_sha256=self.protocol_sha256, normalizer=NORMALIZER, systems=systems, paired=paired, sources=sources,
                      note="Output-conditioned, overlapping descriptive slices; no subgroup confidence intervals or causal/search-error claims. "
                           "Number words include ambiguous 'oh' and ordinals; no semantic number/name classification.")
        Path(self.out_summary.get_path()).write_text(json.dumps(result, indent=2) + "\n")
        lines = ["# Public transcription-convention audit", "", result["note"],
                 "No transcripts are rewritten. WER uses the existing frozen scoring outputs. "
                 "Paired Granite/Parakeet controls are the selected AM-only controls, not necessarily native greedy decoding. "
                 "Additional native controls describe model surface forms: tree lemma orthography can impose capitalization. "
                 "Loquacious total and subsets overlap; no cross-dataset totals are calculated.", ""]
        for dataset in sorted(panels):
            lines += [f"## {dataset}", "", "| System | Units | Cased words | All-caps words | Contractions | Hyphens | Titles | Dotted abbrev. | Digit forms | Number words |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for r in systems:
                if r["dataset"] != dataset:
                    continue
                f = r["occurrences"]
                label = r["family"].replace(" | None | standalone", " control").replace(" | None | native", " native").replace(" | Speech Qwen 1.7B | one-pass", " + Speech 1.7B")
                lines.append("| " + " | ".join(map(str, [label, r["units"]] + [f.get(k, 0) for k in
                    ("uppercase_words", "allcaps_words", "contraction", "hyphen", "title", "dotted_abbreviation", "digit", "number_word")])) + " |")
            for p in paired:
                if p["dataset"] != dataset:
                    continue
                lines += ["", f"### {p['model']} + Speech 1.7B: slices defined by Speech output", "",
                          "| Slice | Units | AM WER | Speech WER | Fusion WER | Δ vs better | Improved units vs S | Worse units vs S |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|"]
                for row in p["slices"]:
                    w = row["wer"]
                    lines.append(f"| {row['slice']} | {row['units']} | {w['am']:.3f} | {w['speech']:.3f} | "
                                 f"{w['fusion']:.3f} | {row['delta_vs_better']:+.3f} | {row['improved_vs_s']} | {row['worsened_vs_s']} |")
        Path(self.out_report.get_path()).write_text("\n".join(lines) + "\n")
