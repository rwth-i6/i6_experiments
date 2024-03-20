__all__ = ["CreateBPELexiconJob"]

import subprocess as sp
import os
import sys
import xml.etree.ElementTree as ET

from sisyphus import *

Path = setup_path(__package__)

from i6_core.lib.lexicon import Lexicon, Lemma
import i6_core.util as util


class CreateBPELexiconJob(Job):
    """
    Create a Bliss lexicon from bpe transcriptions that can be used e.g, for lexicon constrained BPE search.
    """

    def __init__(
        self,
        base_lexicon_path: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: tk.Path,
        subword_nmt_repo: tk.Path,
        unk_label:str = "UNK",
    ):
        """
        :param base_lexicon_path: base lexicon (can be phoneme based) to take the lemmas from
        :param bpe_codes: bpe codes from the ReturnnTrainBPEJob
        :param bpe_vocab: vocab file to limit which bpe splits can be created
        :param subword_nmt_repo: cloned repository
        :param unk_label:
        """
        self.base_lexicon_path = base_lexicon_path
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = subword_nmt_repo
        self.unk_label = unk_label

        self.out_lexicon = self.output_path("lexicon.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        lexicon = Lexicon()

        lm_tokens = set()

        base_lexicon = Lexicon()
        base_lexicon.load(self.base_lexicon_path)
        for l in base_lexicon.lemmata:
            if l.special is None:
                for orth in l.orth:
                    lm_tokens.add(orth)
                for token in l.synt or []:  # l.synt can be None
                    lm_tokens.add(token)
                for eval in l.eval:
                    for t in eval:
                        lm_tokens.add(t)

        lm_tokens = list(lm_tokens)

        with util.uopen("words", "wt") as f:
            for t in lm_tokens:
                f.write(f"{t}\n")

        vocab = set()
        with util.uopen(self.bpe_vocab.get_path(), "rt") as f, util.uopen("fake_count_vocab.txt", "wt") as vocab_file:
            for line in f:
                if "{" in line or "<s>" in line or "</s>" in line or "}" in line:
                    continue
                symbol = line.split(":")[0][1:-1]
                if symbol != self.unk_label:
                    vocab_file.write(symbol + " -1\n")
                    symbol = symbol.replace(".", "_")
                    vocab.add(symbol)
                    lexicon.add_phoneme(symbol.replace(".", "_"))

        apply_binary = os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py")
        args = [
            sys.executable,
            apply_binary,
            "--input",
            "words",
            "--codes",
            self.bpe_codes.get_path(),
            "--vocabulary",
            "fake_count_vocab.txt",
            "--output",
            "bpes",
        ]
        sp.run(args, check=True)

        with util.uopen("bpes", "rt") as f:
            bpe_tokens = [l.strip() for l in f]

        w2b = {w: b for w, b in zip(lm_tokens, bpe_tokens)}

        for w, b in w2b.items():
            b = " ".join([token if token in vocab else self.unk_label for token in b.split()])
            lexicon.add_lemma(Lemma([w], [b.replace(".", "_")]))

        elem = lexicon.to_xml()
        tree = ET.ElementTree(elem)
        util.write_xml(self.out_lexicon.get_path(), tree)
