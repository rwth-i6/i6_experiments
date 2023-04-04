import h5py
import logging
import os
import shutil
import tempfile

from sisyphus import tk, Job, Task

from i6_core.lib.rasr_cache import FileArchive
from i6_core.util import MultiPath


class SprintFeatureToHdf(Job):
    def __init__(self, feature_caches: MultiPath):
        self.feature_caches = feature_caches

        self.out_hdf_files = [
            self.output_path(f"data.hdf.{i + 1}", cached=False) for i in range(len(feature_caches.hidden_paths))
        ]

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1.0}

    def tasks(self):
        yield Task(
            "run",
            rqmt=self.rqmt,
            args=list(self.feature_caches.hidden_paths.keys()),
        )

    def run(self, task_id: int):
        seq_names = []
        string_dt = h5py.special_dtype(vlen=str)

        feature_cache = FileArchive(tk.uncached_path(self.feature_caches.hidden_paths[task_id]))

        with tempfile.TemporaryDirectory() as out_dir:
            out_file = os.path.join(out_dir, "data.hdf")
            logging.info(f"creating HDF at {out_file}")

            with h5py.File(out_file, "w") as out:
                # root
                streams_group = out.create_group("streams")

                # first level
                feature_group = streams_group.create_group("features")
                feature_group.attrs["parser"] = "feature_sequence"

                # second level
                feature_data = feature_group.create_group("data")

                for file in feature_cache.ft:
                    info = feature_cache.ft[file]
                    if info.name.endswith(".attribs"):
                        continue
                    seq_names.append(info.name)

                    # features
                    times, features = feature_cache.read(file, "feat")
                    feature_data.create_dataset(seq_names[-1].replace("/", "\\"), data=features)

                out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)

            target = self.out_hdf_files[task_id - 1].get_path()
            logging.info(f"moving {out_file} to its target {target}")

            shutil.move(out_file, target)
