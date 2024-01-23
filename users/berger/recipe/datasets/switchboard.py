import copy

from sisyphus import *

from i6_core.lib.lexicon import Lexicon, Lemma
from i6_core.util import write_xml


def add_alt_orth_to_lexicon(alt_orth: str, base_orth: str, lexicon: Lexicon):
    """
    Add alternative orthography to all lemmas that contain the base_orth
    """
    for lemma in lexicon.lemmata:
        if base_orth in lemma.orth and alt_orth not in lemma.orth:
            lemma.orth.append(alt_orth)
        lemma.orth.sort(key=str.swapcase)


def combine_lemmata(lemma_1: Lemma, lemma_2: Lemma):
    result = copy.deepcopy(lemma_1)
    result.orth = list(set(result.orth + lemma_2.orth))
    result.orth.sort(key=str.swapcase)
    result.phon = list(set(result.phon + lemma_2.phon))
    result.eval = list(set(result.eval + lemma_2.eval))
    return result


def add_lowercase_orths_to_lexicon(lexicon: Lexicon):
    """
    Add lower-case versions of all orthographies to the lexicon.
    This may introduce overlap, so merge capitalized and lowercase versions
    of the same lemma into one.
    """
    base_lemma_list = []
    lemma_dict = {}
    for lemma in lexicon.lemmata:
        if not lemma.orth or lemma.orth[0].startswith("["):
            base_lemma_list.append(lemma)
            continue
        key_orth = lemma.orth[0].lower()  # Lemmas will be merged if their key_orth is equal
        lowercase_orth = [orth.lower() for orth in lemma.orth]
        lemma.orth = list(set(lemma.orth + lowercase_orth))
        lemma.orth.sort(key=str.swapcase)

        if key_orth in lemma_dict:
            lemma_dict[key_orth] = combine_lemmata(lemma_dict[key_orth], lemma)
        else:
            lemma_dict[key_orth] = lemma

    lexicon.lemmata = base_lemma_list + [lemma_dict[key] for key in sorted(lemma_dict.keys())]


class PreprocessSwitchboardLexiconJob(Job):
    """
    Preprocesses switchboard lexicon by adding differently capitalized versions of orthographies.
    Needed e.g. because language model works with lower-case orths.
    Compare /home/tuske/work/ASR/switchboard/corpus
    """

    def __init__(self, base_lexicon):
        self.base_lexicon = base_lexicon

        self.out_lexicon = self.output_path("lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lexicon_object = Lexicon()
        lexicon_object.load(self.base_lexicon)

        add_lowercase_orths_to_lexicon(lexicon_object)

        for alt_orth, base_orth in [
            ("Act", "act"),
            ("Al", "al"),
            ("Am", "am"),
            ("Are", "are"),
            ("ATs", "ats"),
            ("Bart", "bart"),
            ("BBs", "bbs"),
            ("Bly", "bly"),
            ("Cat", "cat"),
            ("Dart", "dart"),
            ("Dcom", "dcom"),
            ("Deedee", "deedee"),
            ("Epcot", "epcot"),
            ("GTs", "gts"),
            ("In", "in"),
            ("Inc", "inc"),
            ("Kmart", "kmart"),
            ("Kindercare", "kindercare"),
            ("La", "la"),
            ("Lan", "lan"),
            ("Led", "led"),
            ("Ra-", "ra-"),
            ("Seville", "seville"),
            ("Tad", "tad"),
            ("Thirtysomething", "thirtysomething"),
            ("Wasp", "wasp"),
        ]:
            add_alt_orth_to_lexicon(alt_orth, base_orth, lexicon_object)

        write_xml(self.out_lexicon.get_path(), lexicon_object.to_xml())
