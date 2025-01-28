from sisyphus import Job, Task, tk

from i6_core.lib.lexicon import Lexicon, Lemma
from i6_core.util import write_xml


class AddBoundaryMarkerToLexiconJob(Job):

    def __init__(self, bliss_lexicon, add_eow=False, add_sow=False):
        """

        :param tk.Path bliss_lexicon:
        :param bool add_eow:
        :param bool add_sow:
        """
        assert add_eow or add_eow
        self.bliss_lexicon = bliss_lexicon
        self.add_eow = add_eow
        self.add_sow = add_sow

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def modify_phon(self, phon):
        if phon.startswith('[') and phon.endswith(']'):
            return phon
        if self.add_eow:
            phon += '#'
        if self.add_sow and (not self.add_eow or len(phon.split()) > 1):
            phon = '#' + phon
        return phon

    def run(self):

        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()

        for phoneme, variation in in_lexicon.phonemes.items():
            out_lexicon.add_phoneme(phoneme, variation)
            if not (phoneme.startswith("[") or phoneme.endswith("]")):
                out_lexicon.add_phoneme(phoneme + "#", variation)

        for lemma in in_lexicon.lemmata:
            lemma.phon = map(self.modify_phon, lemma.phon)
            out_lexicon.add_lemma(lemma)

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())


class EnsureSilenceFirst(Job):
    """
    Moves the silence phoneme (defined via lemma) to the beginning in the inventory for RASR CTC/Transducer compatibility
    """
    __sis_hash_exclude__ = {'delete_empty_orth': False}

    def __init__(self, bliss_lexicon, delete_empty_orth=False):
        """

        :param tk.Path bliss_lexicon:
        """
        self.bliss_lexicon = bliss_lexicon
        self.delete_empty_orth = delete_empty_orth

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()
        out_lexicon.lemmata = in_lexicon.lemmata

        silence_phon = None
        for lemma in out_lexicon.lemmata:
            if lemma.special == "silence":
                if self.delete_empty_orth:
                    orths = []
                    for orth in lemma.orth:
                        if orth == "":
                            continue
                        orths.append(orth)
                    lemma.orth = orths
                silence_phon = lemma.phon[0]
                assert len(lemma.phon) == 1, (
                    "Silence lemma does not have only one phoneme"
                )
        assert silence_phon, (
            "No silence lemma found"
        )

        out_lexicon.add_phoneme(silence_phon, in_lexicon.phonemes[silence_phon])

        for phoneme, variation in in_lexicon.phonemes.items():
            if phoneme == silence_phon:
                continue
            out_lexicon.add_phoneme(phoneme, variation)

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())

class QuickAndDirtyUpdateLexiconPronunciationsJob(Job):

    def __init__(self, source_lexicon, update_lexicon):
        """

        :param source_lexicon:
        :param update_lexicon:
        """
        self.source_lexicon = source_lexicon
        self.update_lexicon = update_lexicon

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        source_lexicon = Lexicon()
        source_lexicon.load(self.source_lexicon.get_path())

        update_lexicon = Lexicon()
        update_lexicon.load(self.update_lexicon.get_path())

        phon_map = {}
        for lemma in update_lexicon.lemmata:
            # assert len(lemma.orth) <= 1
            for orth in lemma.orth[:1]:
                phon_map[orth] = lemma.phon

        for lemma in source_lexicon.lemmata:
            # assert len(lemma.orth) <= 1
            if len(lemma.orth) == 1:
                if lemma.orth[0] in phon_map.keys():
                    lemma.phon = phon_map[lemma.orth[0]]

        write_xml(self.out_lexicon.get_path(), source_lexicon.to_xml())





