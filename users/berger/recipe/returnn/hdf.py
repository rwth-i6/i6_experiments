__all__ = [
    "BlissCorpusToTargetHdfJob",
    "RemoveBlanksFromAlignmentHdfJob",
    "MatchLengthsJob",
    "FilterHdfSegmentsByLengthRatioJob",
]

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from i6_core.lib import corpus, lexicon
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.util import uopen
from sisyphus import Job, Task, setup_path, tk

assert __package__ is not None
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
        dim: Optional[int] = None,
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
        self.dim = dim

        self.returnn_root = returnn_root

        self.out_hdf = self.output_path("targets.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    @staticmethod
    def _create_target_lookup_dict(lex: lexicon.Lexicon) -> Dict[str, List[int]]:
        # Mapping from phoneme symbol to target index. E.g. {"[SILENCE]": 0, "a": 1, "b": 2, ...}
        phoneme_indices = dict(zip(lex.phonemes.keys(), range(len(lex.phonemes))))

        # build lookup dict of word to target sequence
        lookup_dict: Dict[str, List[int]] = {}
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if not orth:
                    continue
                if len(lemma.phon) > 0:
                    phon = lemma.phon[0]
                else:
                    phon = ""
                lookup_dict[orth] = [phoneme_indices[p] for p in phon.split()]

        return lookup_dict

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

        lookup_dict = self._create_target_lookup_dict(lex)

        if self.word_separation_orth is not None:
            word_separation_targets = lookup_dict[self.word_separation_orth]
            print(
                f"using word separation symbol: {self.word_separation_orth} mapped to targets {word_separation_targets}"
            )
        else:
            word_separation_targets = []

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


class RemoveBlanksFromAlignmentHdfJob(Job):
    """
    Take an alignment and remove all blank labels such that only the non-blanks are left
    """

    def __init__(
        self,
        alignment_hdf: tk.Path,
        blank_label_idx: int,
    ) -> None:
        self.alignment_hdf = alignment_hdf
        self.blank_label_idx = blank_label_idx

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        import h5py
        import numpy as np

        hdf_file = h5py.File(self.alignment_hdf)

        def copy_data_group(src, key, dest):
            src_data = src[key]
            if isinstance(src_data, h5py.Dataset):
                dest.create_dataset(key, data=src_data[:])
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
            if isinstance(src_data, h5py.Group):
                dest.create_group(key)
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
                for sub_key in src_data:
                    copy_data_group(src_data, sub_key, dest[key])

        out_hdf = h5py.File(self.out_hdf, "w")
        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        new_inputs = []
        new_lengths = []
        current_begin_pos = 0
        for tag, length in zip(hdf_file["seqTags"], [length[0] for length in hdf_file["seqLengths"]]):
            full_alignment = hdf_file["inputs"][current_begin_pos : current_begin_pos + length]
            non_blank_labels = full_alignment[full_alignment != self.blank_label_idx]

            new_inputs.extend(non_blank_labels)
            new_length = len(non_blank_labels)
            new_lengths.append([new_length])

            print(f"Reduced alignment of length {length} for sequence {tag} to {new_length} non-blank labels.")
            current_begin_pos += length

        print("Finished processing.")

        matched_inputs = np.array(new_inputs, dtype=hdf_file["inputs"].dtype)
        matched_lengths = np.array(new_lengths, dtype=hdf_file["seqLengths"].dtype)

        out_hdf.create_dataset("inputs", data=matched_inputs)
        for attr_key, attr_val in hdf_file["inputs"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "seqTags", out_hdf)
        out_hdf.create_dataset("seqLengths", data=matched_lengths)
        for attr_key, attr_val in hdf_file["seqLengths"].attrs.items():
            out_hdf["seqLengths"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "targets", out_hdf)


def _identity(x):
    return x


class MatchLengthsJob(Job):
    def __init__(
        self,
        data_hdf: tk.Path,
        match_hdfs: List[tk.Path],
        match_len_transform_func: Optional[Callable[[int], int]] = None,
    ) -> None:
        self.data_hdf = data_hdf
        self.match_hdfs = match_hdfs
        if match_len_transform_func is None:
            self.match_len_transform_func = _identity
        else:
            self.match_len_transform_func = match_len_transform_func

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        import h5py
        import numpy as np

        hdf_file = h5py.File(self.data_hdf)
        match_hdf_files = [h5py.File(match_hdf) for match_hdf in self.match_hdfs]

        def copy_data_group(src, key, dest):
            src_data = src[key]
            if isinstance(src_data, h5py.Dataset):
                dest.create_dataset(key, data=src_data[:])
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
            if isinstance(src_data, h5py.Group):
                dest.create_group(key)
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
                for sub_key in src_data:
                    copy_data_group(src_data, sub_key, dest[key])

        out_hdf = h5py.File(self.out_hdf, "w")
        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        match_length_map = {}
        for match_hdf_file in match_hdf_files:
            match_length_map.update(
                dict(
                    zip(
                        match_hdf_file["seqTags"],
                        [self.match_len_transform_func(length[0]) for length in match_hdf_file["seqLengths"]],
                    )
                )
            )

        matched_inputs = []
        matched_lengths = []
        current_begin_pos = 0
        num_mismatches = 0
        for tag, length in zip(hdf_file["seqTags"], [length[0] for length in hdf_file["seqLengths"]]):
            target_length = match_length_map.get(tag, length)

            if target_length == length:
                matched_seq = hdf_file["inputs"][current_begin_pos : current_begin_pos + length]
            elif length < target_length:
                pad_value = hdf_file["inputs"][current_begin_pos + length - 1]
                pad_list = np.array([pad_value for _ in range(target_length - length)])
                matched_seq = np.concatenate(
                    [hdf_file["inputs"][current_begin_pos : current_begin_pos + length], pad_list], axis=0
                )
                print(
                    f"Length for segment {tag} is shorter ({length}) than the target ({target_length}). Append {pad_list}."
                )
                num_mismatches += 1
            else:  # length > target_length
                print(
                    f"Length for segment {tag} is longer ({length}) than the target ({target_length}). Cut off {hdf_file['inputs'][current_begin_pos + target_length : current_begin_pos + length]}."
                )
                matched_seq = hdf_file["inputs"][current_begin_pos : current_begin_pos + target_length]
                num_mismatches += 1

            assert len(matched_seq) == target_length
            matched_inputs.extend(matched_seq)
            matched_lengths.append([target_length])
            current_begin_pos += length

        print(f"Finished processing. Corrected {num_mismatches} mismatched lengths in total.")

        matched_inputs = np.array(matched_inputs, dtype=hdf_file["inputs"].dtype)
        matched_lengths = np.array(matched_lengths, dtype=hdf_file["seqLengths"].dtype)

        out_hdf.create_dataset("inputs", data=matched_inputs)
        for attr_key, attr_val in hdf_file["inputs"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "seqTags", out_hdf)
        out_hdf.create_dataset("seqLengths", data=matched_lengths)
        for attr_key, attr_val in hdf_file["seqLengths"].attrs.items():
            out_hdf["seqLengths"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "targets", out_hdf)


class FilterHdfSegmentsByLengthRatioJob(Job):
    def __init__(
        self,
        data_hdf: tk.Path,
        comparison_hdfs: List[tk.Path],
        min_ratio: float = 0,
        max_ratio: float = float("inf"),
    ) -> None:
        self.data_hdf = data_hdf
        self.comparison_hdfs = comparison_hdfs
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", rqmt={"cpu": 1, "mem": 16})

    def run(self) -> None:
        import h5py
        import numpy as np

        hdf_file = h5py.File(self.data_hdf)
        comparison_hdf_files = [h5py.File(comparison_hdf) for comparison_hdf in self.comparison_hdfs]

        def copy_data_group(src, key, dest):
            src_data = src[key]
            if isinstance(src_data, h5py.Dataset):
                dest.create_dataset(key, data=src_data[:])
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
            if isinstance(src_data, h5py.Group):
                dest.create_group(key)
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
                for sub_key in src_data:
                    copy_data_group(src_data, sub_key, dest[key])

        out_hdf = h5py.File(self.out_hdf, "w")
        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        comparison_length_map = {}
        for comparison_hdf_file in comparison_hdf_files:
            comparison_length_map.update(
                dict(
                    zip(
                        comparison_hdf_file["seqTags"],
                        [length[0] for length in comparison_hdf_file["seqLengths"]],
                    )
                )
            )

        new_inputs = []
        new_lengths = []
        new_tags = []
        current_begin_pos = 0
        num_too_short = 0
        num_too_long = 0
        for tag, length in zip(hdf_file["seqTags"], [length[0] for length in hdf_file["seqLengths"]]):
            comparison_length = comparison_length_map[tag]

            if length / comparison_length >= self.max_ratio:
                print(
                    f"Sequence {tag} is too long with length {length}, should be at most {self.max_ratio * comparison_length}."
                )
                num_too_long += 1
            elif length / comparison_length <= self.min_ratio:
                print(
                    f"Sequence {tag} is too short with length {length}, should be at lest {self.min_ratio * comparison_length}."
                )
                num_too_short += 1
            else:
                new_inputs.extend(hdf_file["inputs"][current_begin_pos : current_begin_pos + length])
                new_tags.append(tag)
                new_lengths.append([length])

            current_begin_pos += length

        print(
            f"Finished processing. {num_too_short} sequences were too short and {num_too_long} were too long, for a total of {num_too_short + num_too_long} filtered-out sequences."
        )

        new_inputs = np.array(new_inputs, dtype=hdf_file["inputs"].dtype)
        new_lengths = np.array(new_lengths, dtype=hdf_file["seqLengths"].dtype)
        new_tags = np.array(new_tags, dtype=hdf_file["seqTags"].dtype)

        out_hdf.create_dataset("inputs", data=new_inputs)
        for attr_key, attr_val in hdf_file["inputs"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        out_hdf.create_dataset("seqTags", data=new_tags)
        for attr_key, attr_val in hdf_file["seqTags"].attrs.items():
            out_hdf["seqTags"].attrs[attr_key] = attr_val
        out_hdf.create_dataset("seqLengths", data=new_lengths)
        for attr_key, attr_val in hdf_file["seqLengths"].attrs.items():
            out_hdf["seqLengths"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "targets", out_hdf)
