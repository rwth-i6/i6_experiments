import subprocess
import sys
from typing import Dict, List, Optional

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

    def append_st(ph: str, i: int):
        if ph not in result:
            result[ph] = []
        result[ph].append(i)

    for state_sequence in state_sequences:
        cur_mono: Optional[str] = None
        i = 0

        for state in state_sequence:
            if cur_mono is None or cur_mono == state.ph:
                cur_mono = state.ph
                i += 1
            else:
                append_st(state.ph, i)
                i = 1

        append_st(cur_mono, i)

    return result
