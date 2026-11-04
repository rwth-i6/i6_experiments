"""Audited inputs for the pinned OOV tree prototype (never used by old jobs)."""

import json
import math
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter

from i6_core.lib.lexicon import Lexicon
from i6_core.util import uopen, write_xml
from sisyphus import Job, Task, tk

from .huggingface import _get_labels, _load_tokenizer


def classify_piece(token: str, *, special: bool, blank: bool) -> str:
    if blank:
        return "blank"
    if special:
        return "special"
    if not token or any(c.isspace() for c in token) or "▁" in token[1:]:
        raise ValueError(f"Unsupported whitespace/internal boundary in tokenizer piece {token!r}")
    if token == "▁":
        return "separator"
    return "start" if token.startswith("▁") else "internal"


class RemoveEmptyUnknownPronunciationsJob(Job):
    """Keep the LM unknown anchor without introducing an epsilon acoustic path.

    An absent phon element means no pronunciation; an empty phon element defines
    a zero-length pronunciation. The latter reaches CtcTreeBuilder in the
    original-lexicon control, where no fallback inventory exists to exclude it.
    """

    def __init__(self, lexicon: tk.Path):
        self.lexicon = lexicon
        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.lexicon, "rt") as src:
            root = ET.parse(src).getroot()
        for lemma in root.findall("lemma"):
            if lemma.get("special") != "unknown":
                continue
            for phon in list(lemma.findall("phon")):
                if not (phon.text or "").strip():
                    lemma.remove(phon)
        write_xml(self.out_lexicon.get_path(), root)


class InspectWordStartTokenizerJob(Job):
    """Classify every output column and check real dev text without normalizing it."""

    def __init__(self, huggingface_repo_dir: tk.Path, corpus: tk.Path, blank_index: int = 0):
        self.huggingface_repo_dir = huggingface_repo_dir
        self.corpus = corpus
        self.blank_index = blank_index
        self.out_table = self.output_path("pieces.json")
        self.out_report = self.output_path("tokenizer_report.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 1})

    def run(self):
        tokenizer = _load_tokenizer(self.huggingface_repo_dir)
        vocab = tokenizer.get_vocab()
        labels = _get_labels(self.huggingface_repo_dir, self.blank_index, "<blank>")
        by_id = {i: token for token, i in vocab.items()}
        if len(by_id) != len(vocab) or set(by_id) != set(range(len(labels))):
            raise ValueError("Prototype requires a dense, one-to-one tokenizer/model vocabulary")
        rows = []
        for i, label in enumerate(labels):
            token = by_id[i]
            role = classify_piece(token, special=token in tokenizer.all_special_tokens, blank=i == self.blank_index)
            rows.append(
                dict(
                    id=i,
                    label=label,
                    token=token,
                    role=role,
                    punctuation=bool(token.lstrip("▁"))
                    and all(unicodedata.category(c).startswith("P") for c in token.lstrip("▁")),
                )
            )
        if len({r["label"] for r in rows}) != len(rows):
            raise ValueError("Sanitized labels are not injective")
        samples = ["Hello world", "A MURDERER'S CONSCIENCE", "1.5 percent", " hello  world"]
        with uopen(self.corpus, "rt") as src:
            root = ET.parse(src).getroot()
        samples += [orth.text or "" for orth in root.iter("orth")][:200]
        starts = Counter()
        for text in samples:
            pieces = tokenizer.tokenize(text)
            decoded = tokenizer.convert_tokens_to_string(pieces)
            # Byte-fallback pieces spell e.g. '<0xC2><0xA2>', not the decoded '¢'.
            # Only the tokenizer's sequence decoder defines the surface string.
            if decoded != text:
                raise ValueError(f"Tokenizer round-trip mismatch: {text!r} -> {pieces!r} -> {decoded!r}")
            starts["marked" if pieces and pieces[0].startswith("▁") else "unmarked"] += 1
        with uopen(self.out_table, "wt") as out:
            json.dump(rows, out, ensure_ascii=False, indent=2)
        with uopen(self.out_report, "wt") as out:
            json.dump(
                dict(
                    num_labels=len(rows),
                    roles=dict(Counter(r["role"] for r in rows)),
                    sample_count=len(samples),
                    first_piece=dict(starts),
                    punctuation_policy="marker determines boundary; punctuation remains lexical content",
                ),
                out,
                indent=2,
            )


class BuildGraniteOovLexiconJob(Job):
    """One reviewable fallback piece per output label; shared by both search arms.

    Limit ordinary pronunciations to one surface word for this prototype, reporting
    removed multi-boundary variants. Never silently use the unvalidated general case.
    The combined special role for the separator needs the documented parser patch.
    """

    def __init__(self, lexicon: tk.Path, classification: tk.Path):
        self.lexicon = lexicon
        self.classification = classification
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("lexicon_report.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 1})

    def run(self):
        with uopen(self.classification, "rt") as src:
            rows = json.load(src)
        by_label = {r["label"]: r for r in rows}
        with uopen(self.lexicon, "rt") as src:
            root = ET.parse(src).getroot()
        inventory = [p.findtext("symbol") for p in root.find("phoneme-inventory")]
        if inventory != [r["label"] for r in rows]:
            raise ValueError("Acoustic inventory/order differs from model output columns")
        removed = []
        for lemma in root.findall("lemma"):
            if lemma.get("special"):
                continue
            for phon in list(lemma.findall("phon")):
                labels = (phon.text or "").split()
                # Neutral punctuation remains untouched. Ordinary multi-boundary
                # expansions are outside the pinned implementation's tested domain.
                if any(by_label[p]["role"] in {"start", "separator"} for p in labels[1:]):
                    removed.append(dict(orth=lemma.findtext("orth"), pronunciation=phon.text))
                    lemma.remove(phon)
            if not lemma.findall("phon"):
                raise ValueError(f"Single-word restriction removed every pronunciation of {lemma.findtext('orth')!r}")
        existing_orths = {o.text for o in root.iter("orth")}
        for row in rows:
            if row["role"] in {"blank", "special"}:
                continue
            special = "unknown-word-internal" if row["role"] == "internal" else "unknown-word-start"
            if row["role"] == "separator":
                special += "+nonword"
            orth = row["token"] + "&&"
            if orth in existing_orths:
                raise ValueError(f"Diagnostic orthography collision: {orth!r}")
            lemma = ET.SubElement(root, "lemma", special=special)
            ET.SubElement(lemma, "orth").text = orth
            ET.SubElement(lemma, "phon").text = row["label"]
            ET.SubElement(lemma, "synt")
        write_xml(self.out_lexicon.get_path(), root)
        with uopen(self.out_report, "wt") as out:
            json.dump(
                dict(
                    removed_multi_boundary_pronunciations=removed,
                    num_labels=len(inventory),
                    inventory_unchanged=True,
                    scope="single-surface-word pronunciation prototype",
                ),
                out,
                ensure_ascii=False,
                indent=2,
            )


class PrepareOovBoundaryCheckJob(Job):
    """Legacy job retained for existing job serialization; do not use for new checks.

    RASR's text allophone loader skips phonemes starting with '#'. New checks use
    allophones.add-all with zero context instead, covering those phonemes too.
    """

    def __init__(self, lexicon: tk.Path):
        self.lexicon = lexicon
        self.out_allophones = self.output_path("allophones")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(self.lexicon.get_path())
        with uopen(self.out_allophones, "wt") as out:
            out.write("# all four word-boundary flags, one emitting HMM state\n")
            for label in lex.phonemes:
                for flags in ("", "@i", "@f", "@i@f"):
                    out.write(f"{label}{{#+#}}{flags}\n")


class ValidateOovPrototypeInputsJob(Job):
    """Recognition consumes this lexicon only after actual RASR tying/LM validation."""

    __sis_hash_exclude__ = {"pruning_report": None}

    def __init__(
        self,
        lexicon: tk.Path,
        vocab: tk.Path,
        state_tying: tk.Path,
        arpa_lm: tk.Path,
        label_aliases: dict[str, str] | None = None,
        decoder_smoke_report: tk.Path | None = None,
        pruning_report: tk.Path | None = None,
    ):
        self.lexicon, self.vocab, self.state_tying, self.arpa_lm = lexicon, vocab, state_tying, arpa_lm
        self.label_aliases = label_aliases or {}
        self.decoder_smoke_report = decoder_smoke_report
        self.pruning_report = pruning_report
        self.out_lexicon = self.output_path("lexicon.xml.gz")
        self.out_report = self.output_path("validation.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 1})

    def run(self):
        if getattr(self, "pruning_report", None) is not None:
            with uopen(self.pruning_report, "rt") as src:
                if not json.load(src).get("passed"):
                    raise ValueError("Fallback pruning acceptance failed")
        if getattr(self, "decoder_smoke_report", None) is not None:
            with uopen(self.decoder_smoke_report, "rt") as src:
                smoke = json.load(src)
            if smoke.get("passed") is not True or smoke.get("num_cases", 0) < 12:
                raise ValueError("Pinned RASR decoder smoke tests did not pass")
        lex = Lexicon()
        lex.load(self.lexicon.get_path())
        with uopen(self.vocab, "rt") as src:
            labels = [self.label_aliases.get(line.strip(), line.strip()) for line in src if line.strip()]
        if list(lex.phonemes) != labels:
            raise ValueError("Lexicon phoneme order must equal the model output vocabulary")
        unknown = [lemma for lemma in lex.lemmata if lemma.special == "unknown"]
        if len(unknown) != 1 or unknown[0].synt is None or len(unknown[0].synt) != 1:
            raise ValueError(
                f"Expected exactly one singleton unknown with one LM token in {self.lexicon.get_path()}; "
                f"found {len(unknown)} unknown lemmas with synt={[lemma.synt for lemma in unknown]}. "
                "For Granite prototype inputs, construct HuggingFaceBpeBlissLexiconJob with add_unknown_lemma=True."
            )
        unk = unknown[0].synt[0]
        for lemma in lex.lemmata:
            synt = lemma.synt if lemma.synt is not None else (lemma.orth[0].split() if lemma.orth else [])
            if len(synt) > 1:
                raise ValueError(f"Multiple LM tokens on lemma {lemma.orth}: {synt}")
        expected_states = {
            f"{label}{{#+#}}{flags}.0": i for i, label in enumerate(labels) for flags in ("", "@i", "@f", "@i@f")
        }
        tying = {}
        with uopen(self.state_tying, "rt") as src:
            for line in src:
                fields = line.split()
                if not fields:
                    continue
                # A real phoneme may start with '#'. Recognize expected states
                # before applying the comment rule (including indented comments).
                if fields[0] not in expected_states and line.lstrip().startswith("#"):
                    continue
                if len(fields) != 2:
                    raise ValueError(f"Malformed state-tying entry: {line.rstrip()!r}")
                tying[fields[0]] = int(fields[1])
        for state, index in expected_states.items():
            if tying.get(state) != index:
                raise ValueError(f"Incorrect emission index for {state}: {tying.get(state)}, expected {index}")
        unk_log10 = None
        in_unigrams = False
        with uopen(self.arpa_lm, "rt") as src:
            for line in src:
                if line.strip() == "\\1-grams:":
                    in_unigrams = True
                    continue
                if in_unigrams and line.lstrip().startswith("\\"):
                    break
                fields = line.split()
                if in_unigrams and len(fields) >= 2 and fields[1] == unk:
                    unk_log10 = float(fields[0])
        if unk_log10 is None or not math.isfinite(unk_log10) or unk_log10 <= -99:
            raise ValueError(f"ARPA has no usable unigram probability for {unk!r}: {unk_log10}")
        shutil.copyfile(self.lexicon.get_path(), self.out_lexicon.get_path())
        with uopen(self.out_report, "wt") as out:
            json.dump(
                dict(
                    num_labels=len(labels),
                    checked_boundary_states=4 * len(labels),
                    label_aliases=self.label_aliases,
                    unknown_token=unk,
                    unknown_unigram_log10=unk_log10,
                ),
                out,
                indent=2,
            )
