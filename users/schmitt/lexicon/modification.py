import copy
from typing import Optional, List, Dict
from collections import OrderedDict

from sisyphus import Job, Task, tk

from i6_core.lib import lexicon
from i6_core.lib.lexicon import Lemma
from i6_core.util import write_xml


class AddPhonemesAndLemmasToLexiconJob(Job):
    """
    Adds phoneme to lexicon.
    """

    def __init__(
        self, lex_to_modify: tk.Path, phonemes: List[str], variation: Optional[str] = "none", lemmas: List[Dict] = None
    ):
        self.lex_to_modify = lex_to_modify
        self.phonemes = phonemes
        self.variation = variation
        self.lemmas = lemmas

        self.out_lexicon = self.output_path("lex.xml.gz")

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        lex_to_modify = lexicon.Lexicon()
        lex_to_modify.load(self.lex_to_modify.get_path())

        for phoneme in self.phonemes:
            lex_to_modify.add_phoneme(phoneme, self.variation)

        if self.lemmas:
            for lemma in self.lemmas:
                lex_to_modify.add_lemma(Lemma(**lemma))

        write_xml(self.out_lexicon.get_path(), lex_to_modify.to_xml())


class ReorderPhonemeInventoryByReturnnVocabJob(Job):
    """
    Reorders the phoneme inventory of a lexicon to match a returnn (json) vocabulary.

    The job will raise an error if the phoneme set does not match.
    """

    def __init__(self, lex_to_modify: tk.Path, vocab: tk.Path):
        self.lex_to_modify = lex_to_modify
        self.vocab = vocab

        self.out_lexicon = self.output_path("lex.xml.gz")

        self.rqmt = {"cpu": 1, "mem": 1.0, "time": 1.0}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        lex_to_modify = lexicon.Lexicon()
        lex_to_modify.load(self.lex_to_modify.get_path())

        with open(self.vocab, "r") as f:
            vocab = {}
            for line in f:
                if line.startswith("{") or line.startswith("}"):
                    continue
                phoneme, idx = line.split(":")
                # strip away quotes of phoneme and comma behind idx
                vocab[phoneme[1:-1]] = int(idx.strip()[:-1])
            vocab = {v: k for k, v in vocab.items()}

        # breakpoint()

        assert len(vocab) == len(lex_to_modify.phonemes), "Number of phonemes in lexicon and vocab differ!"
        assert set(vocab.values()) == set(lex_to_modify.phonemes.keys())

        old_phonemes = copy.deepcopy(lex_to_modify.phonemes)
        lex_to_modify.phonemes = OrderedDict()
        for idx in range(len(vocab)):
            variation = old_phonemes[vocab[idx]]
            lex_to_modify.add_phoneme(vocab[idx], variation)

        write_xml(self.out_lexicon.get_path(), lex_to_modify.to_xml())
