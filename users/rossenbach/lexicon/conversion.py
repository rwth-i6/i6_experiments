import collections
import gzip
import os.path
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from i6_core.lib.lexicon import Lexicon

from sisyphus import *

Path = setup_path(__package__)

import i6_core.lib.lexicon as lexicon
from i6_core.util import uopen, write_xml


class BlissLexiconToWordLexicon(Job):
    def __init__(self, bliss_lexicon: Path, apply_filter: bool = True):
        self.set_vis_name("Lexicon to Word List")

        self.bliss_lexicon = bliss_lexicon
        self.apply_filter = apply_filter

        self.out_lexicon = self.output_path("lexicon.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        with open(self.out_lexicon.get_path(), "w") as word_file:
            for lemma in lex.lemmata:
                for orth in lemma.orth:
                    for phon in lemma.phon:
                        if len(phon) > 0 and len(orth) > 0:  # we can have empty phonemes or orth
                            word_file.write(f"{orth} {phon}\n")
