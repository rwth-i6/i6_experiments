__all__ = ["FilterCorpusRemoveUnknownWordSegmentsJob", "FilterTextJob"]

import gzip
import xml.etree.cElementTree as ET
from itertools import compress
from typing import List, Callable

from i6_core.lib import corpus
from i6_core.util import uopen

from sisyphus import *

Path = setup_path(__package__)


class FilterCorpusRemoveUnknownWordSegmentsJob(Job):
    """
    Filter segments of a bliss corpus if there are unknowns with respect to a given lexicon
    """

    __sis_hash_exclude__ = {"all_unknown": True}

    def __init__(
        self,
        bliss_corpus: Path,
        bliss_lexicon: Path,
        case_sensitive: bool = False,
        all_unknown: bool = True,
    ):
        """
        :param bliss_corpus:
        :param bliss_lexicon:
        :param case_sensitive: consider casing for check against lexicon
        :param all_unknown: all words have to be unknown in order for the segment to be discarded
        """
        self.corpus = bliss_corpus
        self.lexicon = bliss_lexicon
        self.case_sensitive = case_sensitive
        self.all_unknown = all_unknown

        self.out_corpus = self.output_path("corpus.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        def maybe_to_lower(s):
            return s if self.case_sensitive else s.lower()

        lex_path = self.lexicon.get_path()
        open_func = gzip.open if lex_path.endswith(".gz") else open
        with open_func(lex_path, "rt") as f:
            lex_root = ET.parse(f)
        vocabulary = set([maybe_to_lower(o.text.strip() if o.text else "") for o in lex_root.findall(".//orth")])
        # vocabulary -= {
        #     maybe_to_lower(o.text.strip() if o.text else "")
        #     for l in lex_root.findall(".//lemma")
        #     if l.attrib.get("special") == "unknown"
        #     for o in l.findall(".//orth")
        # }

        c = corpus.Corpus()
        c.load(self.corpus.get_path())

        def unknown_filter(corpus: corpus.Corpus, recording: corpus.Recording, segment: corpus.Segment) -> bool:
            """
            :param corpus: needed to match the filter signature
            :param recording: needed to match the filter signature
            :param segment: segment to filter
            :return: whether the orth of segment contains at least one known word (all_unknown = True) or
                     whether all orths are in the lexicon (all_unknown = False)
            """
            orth = segment.orth
            if not orth:
                return True
            words = [maybe_to_lower(o) for o in orth.strip().split(" ")]

            vocab_mask = [w in vocabulary for w in words]
            inv_vocab_mask = [w not in vocabulary for w in words]

            keep = any(vocab_mask) if self.all_unknown else all(vocab_mask)

            if not keep:
                print(
                    "\nSegment: %s\nOrth: %s\nOov: %s"
                    % (segment.fullname(), segment.orth, list(compress(words, inv_vocab_mask)))
                )

            return keep

        c.filter_segments(unknown_filter)
        c.dump(self.out_corpus.get_path())


# Adapted from Zoltán Tüske's mapRemoveFragSilNoise
def es_filter(text):
    wordMap = {
        "[laughter]-": "[laughter]",
        "[noise]-": "[noise]",
        "[vocalized-noise]-": "[vocalized-noise]",
        "s[laughter]": "[laughter]",
        "<unk>": "",
    }

    text = text.strip()
    words = text.split()
    newwords = []
    for c1, word in enumerate(words):
        if word in wordMap:
            word = wordMap[word]
        if word == "":
            continue
        if word.startswith("[") or word.endswith("]"):
            continue
        if word.startswith("-") or word.endswith("-"):
            continue
        newwords.append(word)
    text = " ".join(newwords)
    return text + "\n"


class FilterTextJob(Job):
    """Filter a text file (i.e. corpus transcription file)"""

    def __init__(self, text_file: Path, filter: Callable[[str], str] = es_filter, gzip: bool = False):
        """
        :param text_file:
        :param gzip:
        """
        self.text_file = text_file
        self.filter = filter
        self.gzip = gzip

        self.out_txt = self.output_path("filtered.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.text_file, "rt") as in_f, uopen(self.out_txt.get_path(), "wt") as out_f:
            for line in in_f:
                out_f.write(self.filter(line))