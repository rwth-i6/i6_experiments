from sisyphus import Job, Task, tk
import enum
import random
import numpy as np

from i6_core.lib.corpus import Corpus, Recording, Segment
from i6_core.lib.lexicon import Lexicon
from i6_core.corpus.transform import ApplyLexiconToCorpusJob

class LexiconStrategy(enum.Enum):
    PICK_FIRST = 0

class ApplyLexiconToTranscriptions(ApplyLexiconToCorpusJob):
    """
        Placeholder only
    """
    def __init__(
        self,
        bliss_corpus,
        bliss_lexicon,
        word_separation_orth,
        strategy=LexiconStrategy.PICK_FIRST,
    ):
        super().__init__(
            bliss_corpus=bliss_corpus,
            bliss_lexicon=bliss_lexicon,
            word_separation_orth=word_separation_orth,
            strategy=strategy
        )


class RandomizeWordOrderJob(Job):
    """

    """
    def __init__(self, bliss_corpus: tk.Path):
        """

        :param bliss_corpus: corpus xml
        """
        self.bliss_corpus = bliss_corpus

        self.out_corpus = self.output_path("random.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = Corpus()
        c.load(self.bliss_corpus.get())
        random.seed(42)

        for r in c.all_recordings():
            for s in r.segments:
                words = s.orth.split()
                random.shuffle(words)
                s.orth = " ".join(words)

        c.dump(self.out_corpus.get())


class FakeCorpusFromLexicon(Job):
    """

    """
    def __init__(self, lexicon: tk.Path, num_sequences, len_min, len_max):
        """

        :param lexicon:
        """
        self.lexicon = lexicon
        self.num_sequences = num_sequences
        self.len_min = len_min
        self.len_max = len_max

        self.out_corpus = self.output_path("lex_random_corpus.xml.gz")

    def run(self):
        lex = Lexicon()
        lex.load(self.lexicon.get())

        rand = np.random.RandomState(seed=42)

        word_list = []
        for lemma in lex.lemmata:
            if lemma.special is not None:
                continue
            for orth in lemma.orth:
                word_list.append(orth)
        word_array = np.asarray(word_list)

        c = Corpus()
        for i in range(self.num_sequences):
            rec = Recording()
            rec.name = "non_rec_%i" % i
            seg = Segment()
            seg.name = "lex_random_seg_%i" % i
            length = rand.randint(low=self.len_min, high=self.len_max + 1)
            seg.orth = " ".join(rand.choice(a=word_array, size=length))
            rec.add_segment(seg)
            c.add_recording(rec)

        c.dump(self.out_corpus.get())

