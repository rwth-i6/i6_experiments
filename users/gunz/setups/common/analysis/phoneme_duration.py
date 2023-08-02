import itertools
import subprocess
import sys
from typing import Dict, List

from i6_core.lib.rasr_cache import FileArchive

from .allophone_state import AllophoneState


def compute_phoneme_durations(cache_file: str, allophones: str) -> Dict[str, List[int]]:
    archive = subprocess.check_output(["cf", cache_file]).decode(sys.stdout.encoding).strip()

    archive = FileArchive(archive)
    archive.setAllophones(allophones)

    result = {}

    files = (file for file in archive.file_list() if not ".attribs" in file)
    alignments = (archive.read(file, "align") for file in files)
    allophone_sequences = ((archive.allophones[t[1]] for t in align) for align in alignments)
    state_sequences = ((AllophoneState.from_alignment_state(st) for st in allos) for allos in allophone_sequences)
    grouped_sequences = (itertools.groupby(st_seq, lambda st: st.ph) for st_seq in state_sequences)

    for groups in grouped_sequences:
        for key, grouped in groups:
            length = sum((1 for _ in grouped))
            if key not in result:
                result[key] = [length]
            else:
                result[key].append(length)

    return result
