from sisyphus import Job, tk, Task
from typing import Callable

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
