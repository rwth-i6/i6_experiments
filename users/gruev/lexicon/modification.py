__all__ = ["AddSpecialLemmataToLexicon"]

from i6_core.lib.lexicon import Lexicon, Lemma
from i6_core.util import write_xml
from sisyphus import tk, Job, Task


class AddSpecialLemmataToLexicon(Job):
    """
    Adds special phonemes and lemmata to files
    """

    def __init__(
        self,
        bliss_lexicon: tk.Path,
    ):
        """
        :param tk.Path bliss_lexicon
        """
        self.bliss_lexicon = bliss_lexicon
        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()

        # Special and corpus-specific phonemes
        out_lexicon.add_phoneme("[SILENCE]", variation="none")
        out_lexicon.add_phoneme("[MUSIC]", variation="none")
        out_lexicon.add_phoneme("[NOISE]", variation="none")
        out_lexicon.phonemes.update(in_lexicon.phonemes)

        # Special and corpus-specific lemmata
        out_lexicon.add_lemma(Lemma(orth=["[MUSIC]", "[music]"], phon=["[MUSIC]"]))
        out_lexicon.add_lemma(Lemma(orth=["[NOISE]", "[noise]"], phon=["[NOISE]"]))

        out_lexicon.add_lemma(
            Lemma(orth=["[SENTENCE-BEGIN]", "[sentence-begin]"], synt=["<s>"], special="sentence-begin")
        )
        out_lexicon.add_lemma(Lemma(orth=["[SENTENCE-END]", "[sentence-end]"], synt=["</s>"], special="sentence-end"))
        out_lexicon.add_lemma(Lemma(orth=["[SILENCE]", "[silence]"], phon=["[SILENCE]"], special="silence"))
        out_lexicon.add_lemma(
            Lemma(orth=["[UNKNOWN]", "[unknown]"], phon=["[NOISE]", "[MUSIC]"], synt=["<unk>"], special="unknown")
        )

        out_lexicon.add_lemma(Lemma(orth=["[VOCALIZED-NOISE]", "[vocalized-noise]"], phon=["[NOISE]"]))
        out_lexicon.add_lemma(Lemma(orth=["[VOCALIZED-UNKNOWN]", "[vocalized-unknown]"], phon=["[NOISE]"]))
        out_lexicon.lemmata += in_lexicon.lemmata

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())
