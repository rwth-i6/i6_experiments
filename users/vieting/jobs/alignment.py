__all__ = ["ExtractWordBoundariesFromAlignmentJob"]

import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, Optional

from sisyphus import Job, Task, Path
from i6_core.lib import corpus, rasr_cache
from i6_core.util import write_xml, MultiPath, uopen


class ExtractWordBoundariesFromAlignmentJob(Job):
    def __init__(self, alignment_cache: MultiPath, corpus_file: Path, allophone_file: Path):
        self.alignment_cache = alignment_cache
        assert isinstance(self.alignment_cache, MultiPath)
        self.concurrent = len(self.alignment_cache.hidden_paths)
        self.corpus_file = corpus_file
        self.allophone_file = allophone_file
        self.out_word_boundaries = self.output_path("word_boundaries.xml.gz", cached=True)

        self.rqmt = {"time": 0.5, "cpu": 1, "mem": 1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        if self.concurrent > 1:
            yield Task("merge", mini_task=True)

    def run(self, task_id: int):
        archive = rasr_cache.FileArchive(self.alignment_cache.hidden_paths[task_id].get())
        archive.setAllophones(self.allophone_file.get_path())
        c = corpus.Corpus()
        c.load(self.corpus_file.get())

        word_boundaries = {}
        for file in archive.ft:
            info = archive.ft[file]
            seq_name = info.name

            if seq_name.endswith(".attribs"):
                continue

            # alignment
            alignment = archive.read(file, "align")
            if not len(alignment):
                continue
            alignment_states = ["%s.%d" % (archive.allophones[t[1]], t[2]) for t in alignment]

            # transcription from corpus
            seg = c.get_segment_by_name(file)
            word_seq = seg.orth.split()

            # get boundaries
            word_boundaries[file] = self.get_boundaries_from_alignment_states(alignment_states, word_seq)

        root = ET.Element("alignment-times")
        tree = ET.ElementTree(root)
        for segment in word_boundaries:
            segment_element = ET.Element("segment", {"name": segment})
            for word, start_frame, end_frame in word_boundaries[segment]:
                word_element = ET.Element("lemma", {"orth": word, "start": str(start_frame), "end": str(end_frame)})
                segment_element.append(word_element)
            root.append(segment_element)

        write_xml(f"word_boundaries.{task_id}.xml.gz", tree)

    def merge(self):
        tree = None
        for task_id in range(1, self.concurrent + 1):
            with gzip.open(f"word_boundaries.{task_id}.xml.gz") as f:
                subtree = ET.parse(f)
            if task_id == 1:
                tree = subtree.getroot()
            else:
                assert tree is not None
                for result in subtree.iter("segment"):
                    tree.append(result)

        write_xml(self.out_word_boundaries.get(), tree)

    @staticmethod
    def get_boundaries_from_alignment_states(alignment: List[str], word_seq: List[str]) -> List[Tuple[str, int, int]]:
        boundaries = []
        word_seq_idx = 0
        start = -1
        prev_phone = "[DUMMY]{#+#}@i@f.0"
        for frame, phone in enumerate(alignment):
            # check for start
            if "[SILENCE]" not in phone and "@i" in phone:
                if (
                    (phone.split(".")[1] < prev_phone.split(".")[-1]) or  # regular phones
                    prev_phone.startswith("[")  # after silence
                ):
                    start = frame
            elif "[" in phone and prev_phone != phone:
                start = frame

            # check for end
            if "[SILENCE]" not in phone and "@f" in phone:
                if phone.split(".")[1] == "2":
                    if frame == len(alignment) - 1 or alignment[frame + 1].split(".")[-1] != "2":
                        end = frame
                        boundaries.append((word_seq[word_seq_idx], start, end))
                        word_seq_idx += 1
            elif "[SILENCE]" in phone and (frame == len(alignment) - 1 or alignment[frame + 1] != phone):
                end = frame
                boundaries.append((phone.split("{")[0], start, end))
            prev_phone = phone
        assert word_seq_idx == len(word_seq), (
            f"Something went wrong, did not match length of word sequence: {word_seq_idx}, {word_seq}"
        )
        return boundaries
