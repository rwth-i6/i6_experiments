__all__ = ["RasrFeaturesToHdf"]

import h5py
import logging
import os
import random
import shutil
import tempfile
import time
import typing

from sisyphus import Job, Path, Task

from i6_core.lib.rasr_cache import FileArchive
from i6_core.util import MultiPath

from ..cache_manager import cache_file


class RasrFeaturesToHdf(Job):
    def __init__(self, feature_caches: typing.Union[MultiPath, typing.List[Path]]):
        self.feature_caches = (
            list(feature_caches.hidden_paths.values()) if isinstance(feature_caches, MultiPath) else feature_caches
        )

        self.out_hdf_files = [self.output_path(f"data.hdf.{i}", cached=False) for i in range(len(self.feature_caches))]
        self.out_single_segment_files = [self.output_path(f"segments.{i}") for i in range(len(self.feature_caches))]

        self.rqmt = {"cpu": 1, "mem": 4, "time": 1.0}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=list(range(len(self.feature_caches))), parallel=10)

    def run(self, *indices: int):
        for index in indices:
            to_sleep = random.randrange(0, 120)
            logging.info(f"sleeping for {to_sleep}s to avoid thundering herd...")
            time.sleep(to_sleep)

            self.process(index)

    def process(self, index: int):
        seq_names = []
        string_dt = h5py.special_dtype(vlen=str)

        logging.info(f"processing {self.feature_caches[index]}")

        cached_path = cache_file(self.feature_caches[index])
        feature_cache = FileArchive(cached_path)

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

            target = self.out_hdf_files[index].get_path()
            logging.info(f"moving {out_file} to its target {target}")

            shutil.move(out_file, target)

        with open(self.out_single_segment_files[index], "wt") as file:
            file.writelines((f"{seq_name.strip()}\n" for seq_name in seq_names))
