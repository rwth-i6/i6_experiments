__all__ = [
    "BlissCorpusToTargetHdfJob",
]

from functools import lru_cache
from typing import Dict, List, Optional, Set
from i6_core.lib import corpus, lexicon
from i6_core.util import uopen

from sisyphus import *

import numpy as np

from i6_core.lib.hdf import get_returnn_simple_hdf_writer

Path = setup_path(__package__)


class BlissCorpusToTargetHdfJob(Job):
    """
    Use a bliss lexicon to convert all words in a bliss corpus into their phoneme representation
    and write these targets to an HDF file.

    Currently only supports picking the first phoneme.
    """

    def __init__(
        self,
        bliss_corpus: tk.Path,
        bliss_lexicon: tk.Path,
        returnn_root: tk.Path,
        segment_file: Optional[tk.Path] = None,
        word_separation_orth: Optional[str] = None,
        blank_index_last: bool = False,
    ):
        """
        :param bliss_corpus: path to a bliss corpus xml
        :param bliss_lexicon: path to a bliss lexicon file
        :param str|None word_separation_orth: a default word separation lemma orth. The corresponding phoneme
            (or phonemes in some special cases) are inserted between each word.
            Usually it makes sense to use something like "[SILENCE]" or "[space]" or so).
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.word_separation_orth = word_separation_orth
        self.segment_file = segment_file
        self.blank_index_last = blank_index_last

        self.returnn_root = returnn_root

        self.out_hdf = self.output_path("targets.hdf")
        self.out_phoneme_indices = self.output_path("phoneme_indices.txt")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)


    def _create_target_lookup_dict(self, lex: lexicon.Lexicon) -> Dict[str, List[int]]:
        # Mapping from phoneme symbol to target index. E.g. {"[SILENCE]": 0, "a": 1, "b": 2, ...}
        if "<blank>" in lex.phonemes.keys() and self.blank_index_last:
            phonemes_without_blank = list(lex.phonemes.keys())
            phonemes_without_blank.remove("<blank>")
            phoneme_indices = dict(zip(phonemes_without_blank, range(len(phonemes_without_blank))))
            phoneme_indices["<blank>"] = len(phonemes_without_blank)
        else:
            phoneme_indices = dict(zip(lex.phonemes.keys(), range(len(lex.phonemes))))
        with open(self.out_phoneme_indices, "w") as f:
            for key in phoneme_indices:
                f.write(f"{key} {phoneme_indices[key]}\n")

        # build lookup dict of word to target sequence
        lookup_dict: Dict[str, List[int]] = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if not orth or not len(lemma.phon):
                    continue
                lookup_dict[orth] = [phoneme_indices[p] for p in lemma.phon[0].split()]

        return lookup_dict

    @lru_cache
    def _get_segment_whitelist(self) -> Optional[Set[str]]:
        # Create whitelist of allowed segments
        if self.segment_file is None:
            return None
        with uopen(self.segment_file, "rt") as f:
            segments_whitelist = set(l.strip() for l in f.readlines() if len(l.strip()) > 0)
        return segments_whitelist

    def _segment_allowed(self, segment_name: str) -> bool:
        whitelist = self._get_segment_whitelist()
        if whitelist is None:
            return True
        return segment_name in whitelist

    def run(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        lookup_dict = self._create_target_lookup_dict(lex)

        if self.word_separation_orth is not None:
            word_separation_targets = lookup_dict[self.word_separation_orth]
            print(
                f"using word separation symbol: {self.word_separation_orth} mapped to targets {word_separation_targets}"
            )
        else:
            word_separation_targets = []

        # Create hdf writer
        out_hdf_writer = get_returnn_simple_hdf_writer(self.returnn_root.get())(filename=self.out_hdf, dim=None)

        # Load corpus
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        # Iterate over corpus segments
        for segment in c.segments():
            # Skip disallowed segments
            if not self._segment_allowed(segment.fullname()):
                continue

            assert segment.orth is not None

            # Create list of targets for each word in the orth
            word_targets = [lookup_dict[word] for word in segment.orth.split()]
            assert len(word_targets) > 0

            # Concatenate all word target lists with the separator targets inserted in between
            segment_targets: List[int] = []
            for word in word_targets[:-1]:
                segment_targets.extend(word)
                segment_targets.extend(word_separation_targets)
            segment_targets.extend(word_targets[-1])

            # Write target sequence into hdf
            out_hdf_writer.insert_batch(
                inputs=np.array(segment_targets).reshape((1, -1)),
                seq_len=[len(segment_targets)],
                seq_tag=[segment.fullname()],
            )
        out_hdf_writer.close()
