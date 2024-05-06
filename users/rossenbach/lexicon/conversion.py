from i6_core.lib.lexicon import Lexicon

from sisyphus import Job, Task, tk


class BlissLexiconToWordLexicon(Job):
    """
    Extract all orth and phone pairs and write them in a line-based file with space separation.
    Will crash deliberately when an orth entry contains spaces.
    """

    def __init__(self, bliss_lexicon: tk.Path, apply_filter: bool = True):
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
                    assert not " " in orth, "orth may not contain spaces"
                    for phon in lemma.phon:
                        if len(phon) > 0 and len(orth) > 0:  # we can have empty phonemes or orth
                            word_file.write(f"{orth} {phon}\n")
