from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from i6_core.lib import corpus, lexicon
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.util import uopen
from sisyphus import Job, Task, setup_path, tk


class BlissCorpusToLangIDJob(Job):
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
        language_id: int,
        token_level: bool = False,
        segment_file: Optional[tk.Path] = None,
        dim: Optional[int] = None,
    ):
        """
        :param bliss_corpus: path to a bliss corpus xml
        :param bliss_lexicon: path to a bliss lexicon file
        """
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon
        self.segment_file = segment_file
        self.dim = dim
        self.language_id = language_id
        self.token_level = token_level

        self.returnn_root = returnn_root

        self.out_hdf = self.output_path("targets.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    @lru_cache
    def _get_segment_whitelist(self) -> Optional[Set[str]]:
        # Create whitelist of allowed segments
        if self.segment_file is None:
            return None
        with uopen(self.segment_file, "rt") as f:
            segments_whitelist = set(line.strip() for line in f.readlines() if len(line.strip()) > 0)
        return segments_whitelist

    def _segment_allowed(self, segment_name: str) -> bool:
        whitelist = self._get_segment_whitelist()
        if whitelist is None:
            return True
        return segment_name in whitelist

    def run(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        # Create hdf writer
        out_hdf_writer = get_returnn_simple_hdf_writer(self.returnn_root.get())(
            filename=self.out_hdf, dim=self.dim, ndim=1
        )

        # Load corpus
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        # Iterate over corpus segments
        for segment in c.segments():
            # Skip disallowed segments
            if not self._segment_allowed(segment.fullname()):
                continue

            assert segment.orth is not None

            # Concatenate all word target lists with the separator targets inserted in between
            if not self.token_level:
                segment_targets = [self.language_id]

            # Write target sequence into hdf
            out_hdf_writer.insert_batch(
                inputs=np.array(segment_targets).reshape((1, -1)),
                seq_len=[len(segment_targets)],
                seq_tag=[segment.fullname()],
            )
        out_hdf_writer.close()