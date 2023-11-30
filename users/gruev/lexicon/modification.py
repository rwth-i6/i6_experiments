__all__ = ["HandleSpecialLemmataInLexicon"]

from typing import List, Optional
from i6_core.lib.lexicon import Lexicon, Lemma
from i6_core.util import write_xml
from sisyphus import tk, Job, Task


class HandleSpecialLemmataInLexicon(Job):
    """
    Adds, modifies or removes special phonemes and lemmata in a bliss lexicon
    """

    def __init__(
        self,
        bliss_lexicon: tk.Path,
        blacklist: Optional[List[str]] = [],
    ):
        """
        :param tk.Path bliss_lexicon
        :param Optional[List[str]] blacklist
        """
        self.bliss_lexicon = bliss_lexicon
        self.blacklist = blacklist
        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()

        # Special and corpus-specific phonemes
        out_lexicon.add_phoneme("[SILENCE]", variation="none")
        # out_lexicon.add_phoneme("[music]", variation="none")
        out_lexicon.add_phoneme("[NOISE]", variation="none")
        out_lexicon.phonemes.update(in_lexicon.phonemes)

        # Remove blacklisted phonemes
        for symbol in self.blacklist:
            out_lexicon.remove_phoneme(symbol)

        # Special lemmata
        # out_lexicon.add_lemma(Lemma(orth=["[music]"], phon=["[MUSIC]"]))
        out_lexicon.add_lemma(Lemma(orth=["[noise]"], phon=["[NOISE]"]))
        out_lexicon.add_lemma(Lemma(orth=["[vocalized-noise]"], phon=["[NOISE]"]))
        out_lexicon.add_lemma(Lemma(orth=["[vocalized-unknown]"], phon=["[NOISE]"]))

        # Corpus-specific lemmata
        out_lexicon.add_lemma(Lemma(orth=["[sentence-begin]"], synt=["<s>"], special="sentence-begin"))
        out_lexicon.add_lemma(Lemma(orth=["[sentence-end]"], synt=["</s>"], special="sentence-end"))
        out_lexicon.add_lemma(Lemma(orth=["[SILENCE]"], phon=["[SILENCE]"], special="silence"))
        out_lexicon.add_lemma(Lemma(orth=["[unknown]"], phon=["[NOISE]"], synt=["<unk>"], special="unknown"))

        # Remove blacklisted lemmata
        out_lexicon.lemmata += [
            lemma
            for lemma in in_lexicon.lemmata
            if not any(symbol in lemma.orth or symbol in lemma.phon for symbol in self.blacklist)
        ]

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())
