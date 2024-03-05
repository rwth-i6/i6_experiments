from sisyphus import Job, tk, Task
from typing import Callable

from i6_core.lib import corpus
from i6_core.lib.corpus import Corpus, Recording, Segment

import h5py


class FilterMismatchedSequencesJob(Job):
    def __init__(
        self,
        feature_hdf: tk.Path,
        target_hdf: tk.Path,
        check_mismatch_func: Callable[[int, int], bool],
        returnn_root: tk.Path,
    ) -> None:
        self.feature_hdf = feature_hdf
        self.target_hdf = target_hdf
        self.mismatch_func = check_mismatch_func
        self.returnn_root = returnn_root

        self.out_segment_blacklist = self.output_path("segment_blacklist")
        self.out_segment_whitelist = self.output_path("segment_whitelist")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        feature_hdf_file = h5py.File(self.feature_hdf)
        target_hdf_file = h5py.File(self.target_hdf)

        segment_blacklist = []
        segment_whitelist = []

        feature_len_dict = dict(zip(list(feature_hdf_file["seqTags"]), list(feature_hdf_file["seqLengths"][:, 0])))

        for tag, target_len in zip(target_hdf_file["seqTags"], target_hdf_file["seqLengths"]):
            if tag not in feature_len_dict:
                print(f"Sequence {tag} is not contained in feature HDF")
                continue
            if self.mismatch_func(feature_len_dict[tag], target_len):
                print(
                    f"Sequence {tag} length mismatch: Feature sequence length is {feature_len_dict[tag]}, target sequence length is {len}"
                )
                segment_blacklist.append(tag)
            else:
                print(f"Sequence {tag} lengths are compatible.")
                segment_whitelist.append(tag)

        with open(self.out_segment_blacklist.get(), "wb") as f:
            f.write(b"\n".join(segment_blacklist))

        with open(self.out_segment_whitelist.get(), "wb") as f:
            f.write(b"\n".join(segment_whitelist))


class FilterCorpusByDurationWordsRatioJob(Job):
    """
    Filter segments based on time/words ratio
    """

    def __init__(self, bliss_corpus, ratio, compressed=True):
        super().__init__()
        self.bliss_corpus = bliss_corpus
        self.ratio = ratio

        self.out_corpus = self.output_path("corpus.xml" + (".gz" if compressed else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        def filter_by_ratio(corpus: Corpus, recording: Recording, segment: Segment) -> bool:
            """
            returns True if T/N >= ratio where T is the duration of the segment in seconds and N is the number of words
            """
            seg_duration = segment.end - segment.start
            num_words = len(segment.orth.strip().split())
            return seg_duration >= self.ratio * num_words

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())
        c.filter_segments(filter_by_ratio)
        c.dump(self.out_corpus.get_path())
