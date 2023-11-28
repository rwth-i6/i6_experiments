__all__ = ["CorpusReplaceOrthFromCallableJob"]

from typing import Optional, Callable
from i6_core.lib import corpus
from sisyphus import *

Path = setup_path(__package__)


class CorpusReplaceOrthFromCallableJob(Job):
    """
    Maps the orth tag according to a function
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
