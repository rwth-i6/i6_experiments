import enum

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
