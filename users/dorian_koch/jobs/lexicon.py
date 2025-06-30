import copy
from collections import OrderedDict, defaultdict
import itertools
from typing import Generator, List, Optional, Set

from sisyphus import Job, Task, tk

from i6_core.lib import lexicon
from i6_core.util import write_xml


class MergeLexiconWithoutDuplicatesJob(Job):
    """
    This is MergeLexiconJob from i6_core, but it filters out duplicates
    """

    def __init__(self, bliss_lexica, sort_phonemes=False, sort_lemmata=False, compressed=True):
        """
        :param list[Path] bliss_lexica: list of bliss lexicon files (plain or gz)
        :param bool sort_phonemes: sort phoneme inventory alphabetically
        :param bool sort_lemmata: sort lemmata alphabetically based on first orth entry
        :param bool compressed: compress final lexicon
        """
        self.lexica = bliss_lexica
        self.sort_phonemes = sort_phonemes
        self.sort_lemmata = sort_lemmata

        self.out_bliss_lexicon = self.output_path("lexicon.xml.gz" if compressed else "lexicon.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        merged_lex = lexicon.Lexicon()

        lexica: list[lexicon.Lexicon] = []
        for lexicon_path in self.lexica:
            lex = lexicon.Lexicon()
            lex.load(lexicon_path.get_path())
            lexica.append(lex)

        # combine the phonemes
        merged_phonemes = OrderedDict()
        for lex in lexica:
            for symbol, variation in lex.phonemes.items():
                if symbol in merged_phonemes.keys():
                    assert variation == merged_phonemes[symbol], "conflicting phoneme variant for phoneme: %s" % symbol
                else:
                    merged_phonemes[symbol] = variation

        if self.sort_phonemes:
            sorted_phoneme_list = [(k, merged_phonemes[k]) for k in sorted(merged_phonemes.keys())]
            for phoneme_tuple in sorted_phoneme_list:
                merged_lex.add_phoneme(symbol=phoneme_tuple[0], variation=phoneme_tuple[1])
        else:
            merged_lex.phonemes = merged_phonemes

        # old code
        """
        # combine the lemmata
        if self.sort_lemmata:
            lemma_dict = defaultdict(list)
            for lex in lexica:
                for lemma in lex.lemmata:
                    # sort by first orth entry
                    orth_key = lemma.orth[0] if lemma.orth else ""
                    lemma_dict[orth_key].append(lemma)
            merged_lex.lemmata = list(itertools.chain(*[lemma_dict[key] for key in sorted(lemma_dict.keys())]))
        else:
            for lex in lexica:
                # check for existing orths to avoid overlap
                merged_lex.lemmata.extend(lex.lemmata)
                """

        lemma_dict: dict[str, lexicon.Lemma] = {}
        num_dups = 0
        for lex in lexica:
            for lemma in lex.lemmata:
                # sort by first orth entry
                orth_key = lemma.orth[0] if lemma.orth else ""
                if orth_key in lemma_dict:
                    assert lemma.orth == lemma_dict[orth_key].orth, f"conflicting lemma for orth: {orth_key}"
                    assert lemma.phon == lemma_dict[orth_key].phon, f"conflicting lemma for orth: {orth_key}"
                    num_dups += 1
                    continue
                    # there are more attributes but i think this is good enough
                lemma_dict[orth_key] = lemma
                if not self.sort_lemmata:
                    merged_lex.lemmata.append(lemma)
        print(f"Number of lemmas: {len(lemma_dict)}")
        print(f"Number of duplicates: {num_dups}")
        
        if self.sort_lemmata:
            merged_lex.lemmata = [lemma_dict[key] for key in sorted(lemma_dict.keys())]

        write_xml(self.out_bliss_lexicon.get_path(), merged_lex.to_xml())
