"""Approach A: corpus-derived candidates, without dev/test vocabulary leakage."""

import json
import time
import xml.etree.ElementTree as ET
from collections import Counter

from i6_core.util import uopen, write_xml
from sisyphus import Job, Task

from .huggingface import _load_tokenizer
from .oov_diagnostics import read_lexicon


class PrepareOovLengthLexiconJob(Job):
    """Gate recognition on preflight; optionally strip fallback, or refresh marks.

    Refreshing BPE final-piece orths is intentional: old cached fallback jobs may
    predate the && marker. The marker never enters syntactic tokens or phonemes.
    """

    __sis_hash_exclude__ = {"pruning_report": None}

    def __init__(
        self, lexicon, correctness_report, margin_report, balance_report, strip_fallback=False, pruning_report=None
    ):
        self.lexicon = lexicon
        self.correctness_report, self.margin_report, self.balance_report = (
            correctness_report,
            margin_report,
            balance_report,
        )
        self.strip_fallback = strip_fallback
        self.pruning_report = pruning_report
        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        if self.pruning_report is not None:
            with uopen(self.pruning_report, "rt") as src:
                if not json.load(src).get("passed"):
                    raise ValueError("Pruning acceptance failed; fallback recognition is blocked")
        with uopen(self.correctness_report, "rt") as src:
            if not json.load(src).get("passed"):
                raise ValueError("Correctness preflight failed")
        with uopen(self.margin_report, "rt") as src:
            if json.load(src)["peak_frame_margin"]["count"] <= 0:
                raise ValueError("Missing acoustic boundary diagnostic")
        with uopen(self.balance_report, "rt") as src:
            if json.load(src)["alpha0"] <= 0:
                raise ValueError("Missing scoring-balance calibration")
        root = read_lexicon(self.lexicon)
        for lemma in list(root.findall("lemma")):
            special = lemma.get("special", "")
            if self.strip_fallback and special.startswith("unknown-"):
                root.remove(lemma)
            elif special == "unknown-final":
                phon = lemma.findall("phon")
                if len(phon) != 1 or len((phon[0].text or "").split()) != 1:
                    raise ValueError("Expected one label per BPE final lemma")
                for orth in lemma.findall("orth"):
                    lemma.remove(orth)
                orth = ET.Element("orth")
                orth.text = phon[0].text + "&&"
                lemma.insert(0, orth)
            elif special == "unknown":
                for phon in list(lemma.findall("phon")):
                    if not (phon.text or "").strip():
                        lemma.remove(phon)
        write_xml(self.out_lexicon.get_path(), root)


class RankOovLexiconCandidatesJob(Job):
    """Rank LM-training words absent from both the recognition lexicon and ARPA.

    No evaluation corpus is accepted as an argument. The list is shared by the
    100k / 300k / full extensions; ties are broken lexicographically.
    """

    def __init__(self, training_text, recognition_lexicon, arpa_lm):
        self.training_text, self.recognition_lexicon, self.arpa_lm = training_text, recognition_lexicon, arpa_lm
        self.out_words = self.output_path("candidates.tsv")
        self.out_report = self.output_path("candidate-report.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 16, "time": 8})

    def run(self):
        root = read_lexicon(self.recognition_lexicon)
        excluded = {o.text.upper() for l in root.findall("lemma") for o in l.findall("orth") if o.text}
        in_unigrams = False
        lm_words = set()
        with uopen(self.arpa_lm, "rt") as src:
            for line in src:
                line = line.strip()
                if line == "\\1-grams:":
                    in_unigrams = True
                    continue
                if in_unigrams and line.startswith("\\"):
                    break
                if in_unigrams and line:
                    fields = line.split()
                    if len(fields) not in (2, 3):
                        raise ValueError(f"Malformed ARPA unigram: {line}")
                    float(fields[0])
                    lm_words.add(fields[1].upper())
        if not lm_words:
            raise ValueError("ARPA vocabulary is empty")
        counts = Counter()
        with uopen(self.training_text, "rt") as src:
            for line in src:
                for word in line.upper().split():
                    if word not in excluded and word not in lm_words:
                        counts[word] += 1
        with uopen(self.out_words, "wt") as dst:
            for word, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
                dst.write(f"{word}\t{count}\n")
        with uopen(self.out_report, "wt") as dst:
            json.dump(
                dict(
                    candidate_types=len(counts),
                    candidate_tokens=sum(counts.values()),
                    arpa_vocabulary_size=len(lm_words),
                    source="LM training text only",
                ),
                dst,
                indent=2,
            )


class ExtendGraniteOovLexiconJob(Job):
    """Add ordinary UNK-token entries without changing any acoustic output index."""

    def __init__(self, lexicon, candidates, tokenizer_table, repo, max_candidates=None):
        if max_candidates is not None and max_candidates < 1:
            raise ValueError(max_candidates)
        self.lexicon, self.candidates, self.tokenizer_table = lexicon, candidates, tokenizer_table
        self.repo, self.max_candidates = repo, max_candidates
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("extension-report.json")
        self.out_rejected = self.output_path("rejected.jsonl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 16, "time": 8})

    def run(self):
        start = time.monotonic()
        root = read_lexicon(self.lexicon)
        inventory = ET.tostring(root.find("phoneme-inventory"))
        if any(l.get("special", "").startswith("unknown-") for l in root.findall("lemma")):
            raise ValueError("Approach A requires a closed lexicon with no fallback lemmas")
        unknown = root.findall('lemma[@special="unknown"]/synt/tok')
        if len(unknown) != 1 or not unknown[0].text:
            raise ValueError("Expected one singleton unknown LM token")
        unk = unknown[0].text
        words = {o.text.upper() for l in root.findall("lemma") for o in l.findall("orth") if o.text}
        with uopen(self.tokenizer_table, "rt") as src:
            table = json.load(src)
        # InspectWordStartTokenizerJob writes a plain, ordered list of rows.
        by_token = {row["token"]: row for row in table}
        tokenizer = _load_tokenizer(self.repo)
        counts = Counter()
        with uopen(self.candidates, "rt") as src, uopen(self.out_rejected, "wt") as rejected:
            for i, line in enumerate(src):
                if self.max_candidates is not None and i >= self.max_candidates:
                    break
                word, frequency = line.rstrip("\n").split("\t")
                counts["considered"] += 1
                if word.upper() in words or len(word.split()) != 1:
                    raise ValueError(f"Non-novel/non-single-word candidate: {word}")
                pronunciations = set()
                for spelling in (word.lower(), word.capitalize(), word.upper()):
                    tokens = tokenizer.tokenize(spelling)
                    rows = [by_token[t] for t in tokens]
                    if (
                        not rows
                        or any(r["role"] in {"special", "blank", "separator"} for r in rows)
                        or any(r["role"] == "start" for r in rows[1:])
                    ):
                        continue
                    pronunciations.add(" ".join(r["label"] for r in rows))
                if not pronunciations:
                    counts["rejected"] += 1
                    rejected.write(
                        json.dumps(
                            dict(
                                word=word,
                                frequency=int(frequency),
                                reason="no representable single-surface-word pronunciation",
                            )
                        )
                        + "\n"
                    )
                    continue
                lemma = ET.SubElement(root, "lemma")
                ET.SubElement(lemma, "orth").text = word
                for pronunciation in sorted(pronunciations):
                    ET.SubElement(lemma, "phon").text = pronunciation
                ET.SubElement(ET.SubElement(lemma, "synt"), "tok").text = unk
                words.add(word.upper())
                counts["added"] += 1
                counts["pronunciations"] += len(pronunciations)
        if ET.tostring(root.find("phoneme-inventory")) != inventory:
            raise AssertionError("Acoustic inventory changed")
        inventory_labels = {p.findtext("symbol") for p in root.findall("phoneme-inventory/phoneme")}
        for lemma in root.findall("lemma"):
            for phon in lemma.findall("phon"):
                if set((phon.text or "").split()) - inventory_labels:
                    raise ValueError(f"Pronunciation references a new acoustic label: {phon.text}")
        write_xml(self.out_lexicon.get_path(), root)
        with uopen(self.out_report, "wt") as dst:
            json.dump(
                dict(
                    counts=counts,
                    unknown_token=unk,
                    max_candidates=self.max_candidates,
                    lexicon_preparation_seconds=time.monotonic() - start,
                    inventory_unchanged=True,
                ),
                dst,
                indent=2,
            )
