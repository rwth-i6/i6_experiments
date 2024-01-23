__all__ = ["CorpusMapOrthJob", "CorpusTextToWordsJob"]

from typing import Callable

from i6_core.lib import corpus
from i6_core.util import uopen
from sisyphus import *

Path = setup_path(__package__)


class CorpusMapOrthJob(Job):
    """
    Maps the orth tag according to a (preprocessing) function
    """

    def __init__(self, bliss_corpus: Path, preprocess_function: Callable[[str], str]):
        """
        :param bliss_corpus: Corpus in which the orth tag is to be replaced
        :param preprocess_function: Mapping to replace orth tag
        """
        self.bliss_corpus = bliss_corpus
        self.preprocess_function = preprocess_function
        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        for s in c.segments():
            s.orth = self.preprocess_function(s.orth)

        c.dump(self.out_corpus.get_path())


class CorpusTextToWordsJob(Job):
    """
    Converts a corpus text file to a words file, one word per line
    """

    def __init__(self, text_file: Path):
        """
        :param text_file:
        """
        self.text_file = text_file
        self.out_txt = self.output_path("words.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        words = set()
        with uopen(self.text_file, "rt") as f:
            for line in f:
                words.update(line.strip().split())

        words = sorted(list(words))
        with uopen(self.out_txt.get_path(), "wt") as f:
            for word in words:
                f.write(word + "\n")