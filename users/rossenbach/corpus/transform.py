import enum

from sisyphus import Job, Task, tk

from i6_core.lib import corpus, lexicon


class LexiconStrategy(enum.Enum):
    PICK_FIRST = 0


class ApplyLexiconToTranscriptions(Job):
    """
    Use a bliss lexicon to convert all words in a bliss lexicon into their phoneme representation

    Currently only supports picking the first phoneme
    """

    def __init__(self, bliss_corpus, bliss_lexicon, word_separation_orth, strategy=LexiconStrategy.PICK_FIRST):
        """

        :param bliss_corpus:
        :param bliss_lexicon:
        :param str word_separation_orth: the default word separation symbol
        :param strategy:
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.word_separation_orth = word_separation_orth
        self.strategy = strategy

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        # build lookup dict
        lookup_dict = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if orth and self.strategy == LexiconStrategy.PICK_FIRST:
                    if len(lemma.phon) > 0:
                        lookup_dict[orth] = lemma.phon[0]

        word_separation_phon = lookup_dict[self.word_separation_orth]
        print("using word separation symbold: %s" % word_separation_phon)
        separator = " %s " % word_separation_phon

        for segment in c.segments():
            try:
                words = [lookup_dict[w] for w in segment.orth.split(" ")]
                segment.orth = separator.join(words)
            except LookupError:
                raise LookupError("Out-of-vocabulary word detected, please make sure that there are no OOVs remaining by e.g. applying G2P")

        c.dump(self.out_corpus.get_path())


