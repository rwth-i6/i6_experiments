from i6_core.lib.lexicon import Lexicon

from sisyphus import Job, Task, tk


class BlissLexiconToWordLexicon(Job):
    def __init__(self, bliss_lexicon: tk.Path):
        self.set_vis_name("Lexicon to Word List")

        self.bliss_lexicon = bliss_lexicon

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
