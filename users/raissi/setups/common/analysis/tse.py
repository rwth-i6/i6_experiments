from collections import Counter
from difflib import SequenceMatcher
import logging
import numpy as np
import subprocess
import shutil
import statistics
from typing import Any, Callable, Dict, Iterator, Optional, List, Tuple, Union, TypeVar
import sys

from sisyphus import Job, Path, Task, tk

from i6_core.lib.rasr_cache import FileArchiveBundle
import i6_core.util as util
import i6_core.lib.rasr_cache as rasr_cache


NUM_TASKS = 16

T = TypeVar("T")

def cache(path: Path) -> str:
    return subprocess.check_output(["cf", path.get_path()]).decode(sys.stdout.encoding).strip()


def get_mixture_indices(a_cache: FileArchiveBundle, segment: str) -> np.ndarray:
    a_data: List[Tuple[int, int, int, int]] = a_cache.read(segment, "align")
    return np.array(a_data)[:, 1] if len(a_data) > 0 else np.array([])  # return only mixture indices


def get_boundaries(a: np.ndarray, silence_idx: int) -> np.ndarray:
    change_mask = np.logical_and(a[1:] != a[:-1], a[1:] != silence_idx)
    boundaries = np.where(change_mask)[0] + 1
    return boundaries


class ComputeTimestampErrorJob(Job):
    """
    A job that computes a time stamp error given a reference alignment.

    The reference alignment can be more fine-grained on the timescale
    than the alignment to be tested.
    """

    __sis_hash_exclude__ = {"fuzzy_match_mismatching_phoneme_sequences": False}

    def __init__(
        self,
        *,
        allophones: Path,
        alignment: Path,
        t_step: float,
        reference_allophones: Path,
        reference_alignment: Path,
        reference_t_step: float,
        fuzzy_match_mismatching_phoneme_sequences: bool = False,
    ):
        assert t_step >= reference_t_step > 0

        super().__init__()

        self.allophones = allophones
        self.alignment = alignment
        self.t_step = t_step

        self.reference_allophones = reference_allophones
        self.reference_alignment = reference_alignment
        self.reference_t_step = reference_t_step

        self.fuzzy_match_mismatching_phoneme_sequences = fuzzy_match_mismatching_phoneme_sequences

        self.out_num_processed = self.output_var("num_processed")
        self.out_num_skipped = self.output_var("num_skipped")
        self.out_tse = self.output_var("out_tse")
        self.out_tse_per_seq = self.output_var("out_tse_per_seq")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        alignment = FileArchiveBundle(cache(self.alignment))
        alignment.setAllophones(self.allophones.get_path())

        ref_alignment = FileArchiveBundle(cache(self.reference_alignment))
        ref_alignment.setAllophones(self.reference_allophones.get_path())

        segments = [f for f in alignment.file_list() if not f.endswith(".attribs")]
        self.out_num_processed.set(len(segments))

        s_idx = next(iter(alignment.files.values())).allophones.index("[SILENCE]{#+#}@i@f")
        ref_s_idx = next(iter(ref_alignment.files.values())).allophones.index("[SILENCE]{#+#}@i@f")

        skipped = 0
        total_dist = 0
        total_num = 0
        tse = {}

        for seg in segments:
            mix_indices = get_mixture_indices(alignment, seg)
            mix_indices_ref = get_mixture_indices(ref_alignment, seg)

            if len(mix_indices) == 0 or len(mix_indices_ref) == 0:
                logging.warning(f"empty alignment for {seg}, skipping")
                skipped += 1
                continue

            begins, ends = self._compute_begins_ends(alignment, seg, mix_indices, s_idx)
            begins_ref, ends_ref = self._compute_begins_ends(ref_alignment, seg, mix_indices_ref, ref_s_idx)

            if len(begins) == len(begins_ref) and len(ends) == len(ends_ref):
                data = [(begins, ends, begins_ref, ends_ref)]
            elif self.fuzzy_match_mismatching_phoneme_sequences:
                skipped += 1
                continue  # for now, impl is difficult

                # Compute matching sequence on decoded allophones to allow mismatches in
                # allophone indices, and then go back to the mixture indices for efficient
                # computation of the boundaries (indexes are the same).
                a_allos = [alignment.files[seg].allophones[mix] for mix in mix_indices]
                b_allos = [ref_alignment.files[seg].allophones[mix] for mix in mix_indices_ref]

                seq_matcher = SequenceMatcher(None, a=a_allos, b=b_allos, autojunk=False)
                match_blocks = [bl for bl in seq_matcher.get_matching_blocks() if bl.size >= 5]

                if len(match_blocks) == 0:
                    logging.info(f"No matching blocks in sequence found. Skipping {seg}.")
                    skipped += 1
                    continue

                logging.info(a_allos)
                logging.info(b_allos)
                logging.info(match_blocks)

                # We do the comparison of the positions on a sequence of the full length to
                # track mismatches in the starts of the blocks.
                #
                # Basically we zero out everything else but the matching block of indices and
                # recompute the begins and ends over that instead.
                mix_indices_all = [np.zeros_like(mix_indices) for _ in match_blocks]
                for mi, bl in zip(mix_indices_all, match_blocks):
                    mi[bl.a : bl.a + bl.size] = mix_indices[bl.a : bl.a + bl.size]
                mix_indices_ref_all = [np.zeros_like(mix_indices_ref) for _ in match_blocks]
                for mi, bl in zip(mix_indices_ref_all, match_blocks):
                    mi[bl.b : bl.b + bl.size] = mix_indices_ref[bl.b : bl.b + bl.size]

                # mix_indices_all = [mix_indices[bl.a : bl.a + bl.size] for bl in match_blocks]
                # mix_indices_ref_all = [mix_indices_ref[bl.b : bl.b + bl.size] for bl in match_blocks]

                data = [
                    (begins, ends, begins_ref, ends_ref)
                    for mix_indices, mix_indices_ref in zip(mix_indices_all, mix_indices_ref_all)
                    for begins, ends in [compute_begins_ends(mix_indices, s_idx)]
                    for begins_ref, ends_ref in [compute_begins_ends(mix_indices_ref, ref_s_idx)]
                ]
            else:
                logging.info(
                    f"len mismatch in {seg} of {len(begins)}/{len(ends)} (alignment) vs. {len(begins_ref)}/{len(ends_ref)} (reference alignment), skipping due to different pronunciation. {len(tse)} alignments already diffed."
                )
                skipped += 1
                continue

            distances = [self._compute_distance(b, e, b_ref, e_ref) for b, e, b_ref, e_ref in data]
            dists, nums = list(zip(*distances))

            total_num += sum(nums)
            total_dist += sum(dists)
            tse[seg] = (sum(dists) / sum(nums)) * self.reference_t_step

        self.out_num_skipped.set(skipped)
        self.out_tse.set((total_dist / total_num) * self.reference_t_step)
        self.out_tse_per_seq.set(tse)

    def _compute_begins_ends(
        self, alignment: FileArchiveBundle, seg_name: str, mix_indices: np.ndarray, silence_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        begins = get_boundaries(mix_indices, silence_idx)
        ends = len(mix_indices) - get_boundaries(mix_indices[::-1], silence_idx)[::-1]

        return begins, ends

    def _compute_distance(
        self, begins: np.ndarray, ends: np.ndarray, ref_begins: np.ndarray, ref_ends: np.ndarray
    ) -> Tuple[float, int]:
        factor = self.t_step / self.reference_t_step
        begins_err = np.abs(begins * factor - ref_begins)
        ends_err = np.abs(ends * factor - ref_ends)
        distance = np.sum(begins_err) + np.sum(ends_err)
        num_boundaries = len(begins) + len(ends)

        return distance, num_boundaries


class ComputeWordLevelTimestampErrorJob(ComputeTimestampErrorJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rqmt["time"] = 4

    def _compute_begins_ends(
        self, alignment: FileArchiveBundle, seg_name: str, mix_indices: np.ndarray, silence_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Find the next phoneme that is at the word-end and "smear" it over
        # the word. This way we consider word-ends for the TSE only.
        #
        # Probably pretty slow this here, but we should be able to avoid separate
        # phoneme sequences this way.
        all_allos = alignment.files[seg_name].allophones
        final_phonemes = [
            next((mix_j for mix_j in mix_indices[i:] if "@f" in all_allos[mix_j]), mix)
            for i, mix in enumerate(mix_indices)
        ]
        return super()._compute_begins_ends(
            alignment=alignment, seg_name=seg_name, mix_indices=np.array(final_phonemes), silence_idx=silence_idx
        )


class ComputeTSEJob(Job):
    """
    Compute TSE of some alignment compared to a reference
    """

    def __init__(
        self,
        alignment_cache: tk.Path,
        ref_alignment_cache: tk.Path,
        allophone_file: tk.Path,
        ref_allophone_file: tk.Path,
        silence_phone: str = "[SILENCE]",
        ref_silence_phone: str = "[SILENCE]",
        upsample_factor: int = 1,
        ref_upsample_factor: int = 1,
        seq_tag_transform: Optional[Callable[[str], str]] = None,
        remove_outlier_limit: Optional[int] = None,
    ) -> None:
        """
        :param alignment_cache: RASR alignment cache file or bundle for which to compute TSEs
        :param ref_alignment_cache: Reference RASR alignment cache file to compare word boundaries to
        :param allophone_file: Allophone file corresponding to `alignment_cache`
        :param ref_allophone_file: Allophone file corresponding to `ref_alignment_cache`
        :param silence_phone: Silence phoneme string in lexicon corresponding to `allophone_file`
        :param ref_silence_phone: Silence phoneme string in lexicon corresponding to `ref_allophone_file`
        :param upsample_factor: Factor to upsample alignment if it was generated by a model with subsampling
        :param ref_upsample_factor: Factor to upsample reference alignment if it was generated by a model with subsampling
        :param seq_tag_transform: Function that transforms seq tag in alignment cache such that it matches the seq tags in the reference
        :param remove_outlier_limit: If set, boundary differences greater than this frame limit are discarded from computation
        """
        self.alignment_cache = alignment_cache
        self.allophone_file = allophone_file
        self.silence_phone = silence_phone
        self.upsample_factor = upsample_factor

        self.ref_alignment_cache = ref_alignment_cache
        self.ref_allophone_file = ref_allophone_file
        self.ref_silence_phone = ref_silence_phone
        self.ref_upsample_factor = ref_upsample_factor

        self.seq_tag_transform = seq_tag_transform
        self.remove_outlier_limit = remove_outlier_limit or float("inf")

        self.out_tse_frames = self.output_var("tse_frames")
        self.out_word_start_frame_differences = self.output_var("start_frame_differences")
        self.out_plot_word_start_frame_differences = self.output_path("start_frame_differences.png")
        self.out_word_end_frame_differences = self.output_var("end_frame_differences")
        self.out_plot_word_end_frame_differences = self.output_path("end_frame_differences.png")
        self.out_boundary_frame_differences = self.output_var("boundary_frame_differences")
        self.out_plot_boundary_frame_differences = self.output_path("boundary_frame_differences.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)
        yield Task("plot", resume="plot", mini_task=True)

    @staticmethod
    def _compute_word_boundaries(
        alignments: Union[rasr_cache.FileArchive, rasr_cache.FileArchiveBundle],
        allophone_map: List[str],
        seq_tag: str,
        silence_phone: str,
        upsample_factor: int,
    ) -> Tuple[List[int], List[int], int]:
        word_starts = []
        word_ends = []

        align_seq = alignments.read(seq_tag, "align")
        assert align_seq is not None

        seq_allophones = [allophone_map[item[1]] for item in align_seq]
        if upsample_factor > 1:
            seq_allophones = sum([[allo] * upsample_factor for allo in seq_allophones], [])

        for t, allophone in enumerate(seq_allophones):
            is_sil = silence_phone in allophone

            if not is_sil and "@i" in allophone and (t == 0 or seq_allophones[t - 1] != allophone):
                word_starts.append(t)

            if (
                not is_sil
                and "@f" in allophone
                and (t == len(seq_allophones) - 1 or seq_allophones[t + 1] != allophone)
            ):
                word_ends.append(t)

        return word_starts, word_ends, len(seq_allophones)

    def run(self) -> None:
        discarded_seqs = 0
        counted_seqs = 0

        start_differences = Counter()
        end_differences = Counter()
        differences = Counter()

        alignments = rasr_cache.open_file_archive(self.alignment_cache.get())
        alignments.setAllophones(self.allophone_file.get())
        if isinstance(alignments, rasr_cache.FileArchiveBundle):
            allophone_map = next(iter(alignments.archives.values())).allophones
        else:
            allophone_map = alignments.allophones

        ref_alignments = rasr_cache.open_file_archive(self.ref_alignment_cache.get())
        ref_alignments.setAllophones(self.ref_allophone_file.get())
        if isinstance(ref_alignments, rasr_cache.FileArchiveBundle):
            ref_allophone_map = next(iter(ref_alignments.archives.values())).allophones
        else:
            ref_allophone_map = ref_alignments.allophones

        file_list = [tag for tag in alignments.file_list() if not tag.endswith(".attribs")]

        for idx, seq_tag in enumerate(file_list, start=1):
            word_starts, word_ends, seq_length = self._compute_word_boundaries(
                alignments, allophone_map, seq_tag, self.silence_phone, self.upsample_factor
            )
            assert len(word_starts) == len(
                word_ends
            ), f"Found different number of word starts ({len(word_starts)}) than word ends ({len(word_ends)}). Something seems to be broken."

            if self.seq_tag_transform is not None:
                ref_seq_tag = self.seq_tag_transform(seq_tag)
            else:
                ref_seq_tag = seq_tag

            ref_word_starts, ref_word_ends, ref_seq_length = self._compute_word_boundaries(
                ref_alignments, ref_allophone_map, ref_seq_tag, self.ref_silence_phone, self.ref_upsample_factor
            )
            assert len(ref_word_starts) == len(
                ref_word_ends
            ), f"Found different number of word starts ({len(word_starts)}) than word ends ({len(word_ends)}) in reference. Something seems to be broken."

            if len(word_starts) != len(ref_word_starts):
                print(
                    f"Sequence {seq_tag} ({idx} / {len(file_list)}:\n    Discarded because the number of words in alignment ({len(word_starts)}) does not equal the number of words in reference ({len(ref_word_starts)})."
                )
                discarded_seqs += 1
                continue

            shorter_seq_length = min(seq_length, ref_seq_length)
            word_starts = [min(start, shorter_seq_length) for start in word_starts]
            word_ends = [min(end, shorter_seq_length) for end in word_ends]
            ref_word_starts = [min(start, shorter_seq_length) for start in ref_word_starts]
            ref_word_ends = [min(end, shorter_seq_length) for end in ref_word_ends]

            seq_word_start_diffs = [start - ref_start for start, ref_start in zip(word_starts, ref_word_starts)]
            seq_word_start_diffs = [diff for diff in seq_word_start_diffs if abs(diff) <= self.remove_outlier_limit]
            seq_word_end_diffs = [end - ref_end for end, ref_end in zip(word_ends, ref_word_ends)]
            seq_word_end_diffs = [diff for diff in seq_word_end_diffs if abs(diff) <= self.remove_outlier_limit]
            seq_differences = seq_word_start_diffs + seq_word_end_diffs

            start_differences.update(seq_word_start_diffs)
            end_differences.update(seq_word_end_diffs)
            differences.update(seq_differences)

            if seq_differences:
                seq_tse = statistics.mean(abs(diff) for diff in seq_differences)

                print(
                    f"Sequence {seq_tag} ({idx} / {len(file_list)}):\n    Word start distances are {seq_word_start_diffs}\n    Word end distances are {seq_word_end_diffs}\n    Sequence TSE is {seq_tse} frames"
                )
                counted_seqs += 1
            else:
                print(
                    f"Sequence {seq_tag} ({idx} / {len(file_list)}):\n    Discarded since all distances are over the upper limit"
                )
                discarded_seqs += 1
                continue

        print(
            f"Processing finished. Computed TSE value based on {counted_seqs} sequences; {discarded_seqs} sequences were discarded."
        )

        self.out_word_start_frame_differences.set(
            {key: start_differences[key] for key in sorted(start_differences.keys())}
        )
        self.out_word_end_frame_differences.set({key: end_differences[key] for key in sorted(end_differences.keys())})
        self.out_boundary_frame_differences.set({key: differences[key] for key in sorted(differences.keys())})
        self.out_tse_frames.set(statistics.mean(abs(diff) for diff in differences.elements()))

    def plot(self):
        for descr, dict_file, plot_file in [
            (
                "start",
                self.out_word_start_frame_differences.get_path(),
                self.out_plot_word_start_frame_differences.get_path(),
            ),
            (
                "end",
                self.out_word_end_frame_differences.get_path(),
                self.out_plot_word_end_frame_differences.get_path(),
            ),
            (
                "boundary",
                self.out_boundary_frame_differences.get_path(),
                self.out_plot_boundary_frame_differences.get_path(),
            ),
        ]:
            with open(dict_file, "r") as f:
                diff_dict = eval(f.read())

            ranges = [-30, -20, -15, -10, -5, -1, 2, 6, 11, 16, 21, 31]

            range_strings = []
            range_strings.append(f"<{ranges[0]}")
            for idx in range(1, len(ranges)):
                range_strings.append(f"{ranges[idx - 1]} - {ranges[idx] - 1}")
            range_strings.append(f">{ranges[-1] - 1}")

            range_counts = [0] * (len(ranges) + 1)

            for key, count in diff_dict.items():
                idx = 0
                while idx < len(ranges) and ranges[idx] <= key:
                    idx += 1

                range_counts[idx] += count

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')

            plt.figure(figsize=(10, 6))
            plt.bar(range_strings, range_counts, color="skyblue")
            plt.xlabel(f"Word {descr} shift (frames)")
            plt.ylabel("Counts")
            plt.title(f"Word {descr} shift counts")
            plt.xticks(rotation=45)

            plt.savefig(plot_file)
