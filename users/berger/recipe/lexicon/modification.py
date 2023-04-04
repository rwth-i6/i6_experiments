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
        if phon.startswith("[") and phon.endswith("]"):
            return phon
        if phon.startswith("<") and phon.endswith(">"):
            return phon
        if self.add_eow:
            phon += "#"
        if self.add_sow and (not self.add_eow or len(phon.split()) > 1):
            phon = "#" + phon
        return phon

    def run(self):

        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()

        eow_phon_list = []
        sow_phon_list = []
        for phoneme, variation in in_lexicon.phonemes.items():
            out_lexicon.add_phoneme(phoneme, variation)
            if not (
                phoneme.startswith("[")
                or phoneme.endswith("]")
                or phoneme.startswith("<")
                or phoneme.endswith(">")
            ):
                if self.add_eow:
                    eow_phon_list.append((phoneme + "#", variation))
                if self.add_sow:
                    sow_phon_list.append(("#" + phoneme, variation))

        for eow_phon, variation in eow_phon_list:
            out_lexicon.add_phoneme(eow_phon, variation)

        for sow_phon, variation in sow_phon_list:
            out_lexicon.add_phoneme(sow_phon, variation)

        for lemma in in_lexicon.lemmata:
            lemma.phon = map(self.modify_phon, lemma.phon)
            out_lexicon.add_lemma(lemma)

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())


class DeleteEmptyOrthJob(Job):
    """
    Removes the empty pronunciation from the silence lemma of the given lexicon. This avoids unneeded blank insertions in CTC automaton
    """

    def __init__(self, bliss_lexicon: tk.Path):
        self.bliss_lexicon = bliss_lexicon

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()
        out_lexicon.phonemes = in_lexicon.phonemes
        out_lexicon.lemmata = in_lexicon.lemmata

        silence_phon = None
        for lemma in out_lexicon.lemmata:
            if lemma.special == "silence":
                orths = []
                for orth in lemma.orth:
                    if orth == "":
                        continue
                    orths.append(orth)
                lemma.orth = orths
                silence_phon = lemma.phon[0]
                assert (
                    len(lemma.phon) == 1
                ), "Silence lemma does not have only one phoneme"
        assert silence_phon, "No silence lemma found"

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())


class EnsureSilenceFirstJob(Job):
    """
    Moves the silence phoneme (defined via lemma) to the beginning in the inventory for RASR CTC/Transducer compatibility
    """

    def __init__(self, bliss_lexicon: tk.Path):
        """

        :param tk.Path bliss_lexicon:
        """
        self.bliss_lexicon = bliss_lexicon

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
                silence_phon = lemma.phon[0]
                assert (
                    len(lemma.phon) == 1
                ), "Silence lemma does not have only one phoneme"
        assert silence_phon, "No silence lemma found"

        out_lexicon.add_phoneme(silence_phon, in_lexicon.phonemes[silence_phon])

        for phoneme, variation in in_lexicon.phonemes.items():
            if phoneme == silence_phon:
                continue
            out_lexicon.add_phoneme(phoneme, variation)

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())


class EnsureUnknownPronunciationOrthJob(Job):
    """
    Ensures that the unknown lemma has a pronunciation.
    """

    def __init__(self, bliss_lexicon: tk.Path):
        """
        :param tk.Path bliss_lexicon: input lexicon
        """
        self.bliss_lexicon = bliss_lexicon

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()
        out_lexicon.phonemes = in_lexicon.phonemes
        out_lexicon.lemmata = in_lexicon.lemmata

        silence_phon = None
        for lemma in out_lexicon.lemmata:
            if lemma.special == "silence":
                silence_phon = lemma.phon[0]
                assert (
                    len(lemma.phon) == 1
                ), "Silence lemma does not have only one phoneme"
                break
        assert silence_phon, "No silence lemma found"

        for lemma in out_lexicon.lemmata:
            if lemma.special == "unknown":
                if not lemma.phon:
                    lemma.phon.append(silence_phon)
                break

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())


class MakeBlankLexiconJob(Job):
    """
    Modified a bliss lexicon to make it compatible for RASR CTC graph building.
    Adds a <blank> phoneme and lemma. If <separate_silence> is false, additionally
    remove empty orth from silence lemma, remove silence phone and replace all instances of
    the silence phone in lemmata with blank.
    """

    def __init__(self, bliss_lexicon: tk.Path, separate_silence: bool = False) -> None:
        """
        :param tk.Path bliss_lexicon: input lexicon
        """
        self.bliss_lexicon = bliss_lexicon
        self.separate_silence = separate_silence

        self.out_lexicon = self.output_path("lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        blank_phon = "<blank>"

        in_lexicon = Lexicon()
        in_lexicon.load(self.bliss_lexicon.get_path())

        out_lexicon = Lexicon()
        out_lexicon.add_phoneme(blank_phon, variation="none")
        out_lexicon.phonemes.update(in_lexicon.phonemes)

        out_lexicon.add_lemma(
            Lemma(orth=[blank_phon], phon=[blank_phon], special="blank")
        )
        out_lexicon.lemmata += in_lexicon.lemmata

        if not self.separate_silence:
            silence_phon = None
            for lemma in out_lexicon.lemmata:
                if lemma.special == "silence":
                    # Extract silence phone
                    silence_phon = lemma.phon[0]
                    assert (
                        len(lemma.phon) == 1
                    ), "Silence lemma does not have only one phoneme"

                    # Remove empty orth from silence lemma
                    orths = []
                    for orth in lemma.orth:
                        if orth == "":
                            continue
                        orths.append(orth)
                    lemma.orth = orths

                    break

            if silence_phon is not None:
                out_lexicon.remove_phoneme(silence_phon)

                # Replace all occurrences of silence_phon with blank_phon
                for lemma in out_lexicon.lemmata:
                    phons = []
                    for phon in lemma.phon:
                        phons.append(phon.replace(silence_phon, blank_phon))
                    lemma.phon = phons

        write_xml(self.out_lexicon.get_path(), out_lexicon.to_xml())
