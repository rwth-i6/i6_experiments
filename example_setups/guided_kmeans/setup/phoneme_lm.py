"""Phonemized LM text with silence, for the count LMs in ``librasr_recognition.phonetic_lm_dict``.

The phoneme count LMs (``phon_count_{2..9}gram``) were built in a colleague's setup
from ``TextToPhonemeJob`` of the ``haotian_lm`` pipeline
(``setup/external/haotian_lm/data/phon_lm.py``, removed from this tree after commit
f0b150ad3), with ``SilenceModel(between_word=0.1, sentence_bounds=0.1)``: ``<SIL>``
inserted at random, with one probability for both sentence ends. This is that job,
so that LMs with other silence statistics can be built from the same inputs, with two
changes:

* the sentence begin and end take separate probabilities (``sentence_begin`` /
  ``sentence_end``; None falls back to ``sentence_bounds``). The random draws happen
  in the original order, so a model that leaves them None reproduces the original
  text line for line;
* the job reports how much silence it inserted (``out_silence_statistics``).
"""

import itertools as it
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from sisyphus import Job, Task, tk

from i6_core.util import uopen


def _load_lexicon_word_to_phon(bliss_lexicon_path: str) -> Dict[str, str]:
    """Word to first pronunciation from a Bliss lexicon; special lemmata are skipped."""
    word_to_phon: Dict[str, str] = {}
    with uopen(bliss_lexicon_path, "rt") as f:
        tree = ET.parse(f)
    for lemma in tree.findall(".//lemma"):
        if lemma.get("special") is not None:
            continue
        orth_el = lemma.find("orth")
        phon_el = lemma.find("phon")
        if orth_el is None or phon_el is None or orth_el.text is None or phon_el.text is None:
            continue
        word = orth_el.text.strip()
        if not word or word in word_to_phon:
            continue
        word_to_phon[word] = phon_el.text.strip()
    return word_to_phon


def _load_g2p_lexicon(g2p_lexicon_path: str) -> Dict[str, str]:
    """Word to first pronunciation from a Sequitur G2P output (``ApplyG2PModelJob``)."""
    word_to_phon: Dict[str, str] = {}
    with uopen(g2p_lexicon_path, "rt", encoding="utf-8") as f:
        for orth, data in it.groupby(
            (line.strip().split("\t") for line in f if line.strip()),
            lambda t: t[0],
        ):
            for entry in data:
                if len(entry) == 4 and orth not in word_to_phon:
                    word_to_phon[orth] = entry[3].strip()
                    break
    return word_to_phon


@dataclass(frozen=True)
class SilenceModel:
    """Where ``silence_token`` is inserted into the phonemized text, at random.

    :param between_word: probability after each word but the last
    :param sentence_bounds: probability at the sentence begin and at its end, each
        drawn on its own
    :param sentence_begin: overrides ``sentence_bounds`` at the begin
    :param sentence_end: overrides ``sentence_bounds`` at the end
    """

    silence_token: str = "<SIL>"
    between_word: float = 0.0
    sentence_bounds: float = 0.0
    random_seed: int = 1307
    sentence_begin: Optional[float] = None
    sentence_end: Optional[float] = None

    @property
    def begin(self) -> float:
        return self.sentence_bounds if self.sentence_begin is None else self.sentence_begin

    @property
    def end(self) -> float:
        return self.sentence_bounds if self.sentence_end is None else self.sentence_end


def phonemize_lines(
    lines: Iterable[str],
    word_to_phon: Dict[str, str],
    silence_model: Optional[SilenceModel],
    counts: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[Optional[str], List[str]]]:
    """``TextToPhonemeJob``'s per-line work, one result per non-empty input line.

    Yields ``(text, unresolved)``: the phonemized line, or None if a word has no
    pronunciation (the line is dropped), and the words that had none. A dropped
    line still consumes its random draws, as the original job's did, so that the
    lines after it come out the same. ``counts``, if given, is incremented with the
    silence of the lines that are kept.
    """
    sm = silence_model or SilenceModel()
    rng = random.Random(sm.random_seed)
    for line in lines:
        words = line.split()
        if not words:
            continue
        phons: List[str] = []
        unresolved: List[str] = []
        begin = rng.random() < sm.begin
        if begin:
            phons.append(sm.silence_token)
        between = 0
        for i, word in enumerate(words):
            phon = word_to_phon.get(word)
            if phon is None:
                unresolved.append(word)
                break
            phons.append(phon)
            if i < len(words) - 1 and rng.random() < sm.between_word:
                phons.append(sm.silence_token)
                between += 1
        end = rng.random() < sm.end
        if end:
            phons.append(sm.silence_token)
        if unresolved:
            yield None, unresolved
            continue
        if counts is not None:
            counts["sentences"] += 1
            counts["silence_begin"] += int(begin)
            counts["silence_end"] += int(end)
            counts["silence_between_word"] += between
            counts["word_boundaries"] += len(words) - 1
            counts["phonemes"] += sum(len(p.split()) for p in phons) - begin - end - between
        yield " ".join(phons), []


class TextToPhonemeJob(Job):
    """
    Tokenize plain text into whitespace-separated phoneme sequences using a Bliss
    lexicon (first pronunciation per word) plus an optional G2P fallback for OOVs,
    inserting silence as ``silence_model`` says. Lines that still contain
    unresolvable words are dropped.
    """

    def __init__(
        self,
        text_file: tk.Path,
        bliss_lexicon: tk.Path,
        g2p_lexicon: Optional[tk.Path] = None,
        gzip_output: bool = True,
        silence_model: Optional[SilenceModel] = None,
    ):
        super().__init__()
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.gzip_output = gzip_output
        self.silence_model = silence_model

        out_name = "phon.txt.gz" if gzip_output else "phon.txt"
        self.out_text = self.output_path(out_name)
        self.out_unresolved = self.output_path("unresolved_words.txt")
        self.out_num_sentences_in = self.output_var("num_sentences_in")
        self.out_num_sentences_out = self.output_var("num_sentences_out")
        self.out_num_unresolved = self.output_var("num_unresolved")
        #: Counts over the kept sentences, and the rates they give: per sentence
        #: begin / end, per word boundary, and silence tokens per 100 phonemes.
        self.out_silence_statistics = self.output_path("silence_statistics.json")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        word_to_phon = _load_lexicon_word_to_phon(self.bliss_lexicon.get_path())
        if self.g2p_lexicon is not None:
            for word, phon in _load_g2p_lexicon(self.g2p_lexicon.get_path()).items():
                word_to_phon.setdefault(word, phon)

        counts = dict.fromkeys(("sentences", "silence_begin", "silence_end",
                                "silence_between_word", "word_boundaries", "phonemes"), 0)
        unresolved_words: set = set()
        n_in = 0
        n_out = 0
        with uopen(self.text_file.get_path(), "rt") as inf, \
                uopen(self.out_text.get_path(), "wt") as outf:

            def lines():
                nonlocal n_in
                for line in inf:
                    n_in += 1
                    yield line

            for text, unresolved in phonemize_lines(lines(), word_to_phon, self.silence_model,
                                                    counts):
                unresolved_words.update(unresolved)
                if text is None:
                    continue
                outf.write(text + "\n")
                n_out += 1

        with uopen(self.out_unresolved.get_path(), "wt") as f:
            for word in sorted(unresolved_words):
                f.write(word + "\n")
        self.out_num_sentences_in.set(n_in)
        self.out_num_sentences_out.set(n_out)
        self.out_num_unresolved.set(len(unresolved_words))

        sentences = max(counts["sentences"], 1)
        silence = counts["silence_begin"] + counts["silence_end"] + counts["silence_between_word"]
        rates = {
            "sentence_begin": counts["silence_begin"] / sentences,
            "sentence_end": counts["silence_end"] / sentences,
            "between_word": counts["silence_between_word"] / max(counts["word_boundaries"], 1),
            "per_100_phonemes": 100 * silence / max(counts["phonemes"], 1),
        }
        with open(self.out_silence_statistics.get_path(), "w") as f:
            json.dump({"counts": counts, "rates": rates}, f, indent=2)
