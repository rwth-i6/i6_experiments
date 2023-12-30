__all__ = ["SegmentCorpusNoPrefixJob"]

import collections
import itertools as it
import os
import random
import re

import numpy as np

from i6_core.util import MultiOutputPath
from i6_core.lib import corpus
from i6_core.util import chunks, uopen

from sisyphus import *



class SegmentCorpusNoPrefixJob(Job):
    def __init__(self, bliss_corpus, num_segments, remove_prefix):
        self.set_vis_name("Segment Corpus")

        self.bliss_corpus = bliss_corpus
        self.num_segments = num_segments
        self.remove_prefix = remove_prefix
        self.out_single_segment_files = dict(
            (i, self.output_path("segments.%d" % i)) for i in range(1, num_segments + 1)
        )
        self.out_segment_path = MultiOutputPath(self, "segments.$(TASK)", self.out_single_segment_files)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        all_segments = list(c.segments())

        for idx, segments in enumerate(chunks(all_segments, self.num_segments)):
            with open(self.out_single_segment_files[idx + 1].get_path(), "wt") as segment_file:
                for segment in segments:
                    name = segment.fullname() + "\n"
                    name = name[name.startswith(self.remove_prefix) and len(self.remove_prefix):]
                    segment_file.write(name)

    @classmethod
    def hash(cls, kwargs):
        d = dict(**kwargs)
        if not d["remove_prefix"]:
            d.pop("remove_prefix")
        return super().hash(d)