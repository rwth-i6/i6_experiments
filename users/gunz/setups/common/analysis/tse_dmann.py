import numpy as np
from typing import Optional

from sisyphus import Job, Path, Task

from i6_core.lib import rasr_cache

from .tse import cache


class DMannComputeTseJob(Job):
    def __init__(
        self, alignment: Path, reference_alignment: Path, allophones: Path, segment_file: Optional[Path] = None
    ):
        self.alignment = alignment
        self.reference_alignment = reference_alignment
        self.allophones = allophones
        self.segment_file = segment_file

        self.out_tse = self.output_var("tse")

        self.rqmts = {"time": 1, "cpu": 1, "mem": 8}

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        # load archives
        archives = []
        for a in [self.alignment, self.reference_alignment]:
            archive = rasr_cache.open_file_archive(cache(a))
            archive.setAllophones(self.allophones.get_path())
            archives.append(archive)

        segments = (
            list(archives[1].file_list())
            if not self.segment_file
            else [l.strip() for l in open(self.segment_file.get_path())]
        )

        # silence allophone idx
        some_archive = list(archives[1].archives.values())[0]
        sidx = some_archive.allophones.index("[SILENCE]{#+#}@i@f")

        def get_phoneme_boundaries(allos):
            change_mask = np.logical_and(allos[1:] != allos[:-1], allos[1:] != sidx)
            boundary_idxs = np.where(change_mask)[0] + 1
            return boundary_idxs

        total_distance = total_n_boundaries = 0
        for seq_tag in segments:
            if seq_tag.endswith(".attribs"):
                continue
            phoneme_starts = []
            phoneme_ends = []
            for av in archives:
                align = av.read(seq_tag, "align")
                align = np.array([a[1] for a in align])
                if len(align) == 0:
                    break
                assert len(align) > 0

                phoneme_starts.append(get_phoneme_boundaries(align))
                phoneme_ends.append(len(align) - get_phoneme_boundaries(align[::-1])[::-1])

            if len(phoneme_starts) < 2:
                continue

            if len(phoneme_starts[0]) != len(phoneme_starts[1]):
                continue

                # error handling
                allos = []
                for av in archives:
                    align = av.read(seq_tag, "align")
                    align = np.array([a[1] for a in align])
                    allos.append(list(map(lambda s: some_archive.allophones[s], align)))

                assert False, "Different number of phonemes: %s vs %s for sequences %s, %s" % (
                    len(phoneme_starts[0]),
                    len(phoneme_starts[1]),
                    *allos,
                )
            if len(phoneme_ends[0]) != len(phoneme_ends[1]):
                continue

            assert len(phoneme_starts[0]) == len(phoneme_starts[1])

            distances = np.concatenate((np.abs(np.subtract(*phoneme_starts)), np.abs(np.subtract(*phoneme_ends)))).sum()

            total_distance += distances
            total_n_boundaries += len(phoneme_starts[0]) * 2

        tse = total_distance / total_n_boundaries
        self.out_tse.set(tse)
