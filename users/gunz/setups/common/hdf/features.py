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

from i6_core.lib.rasr_cache import FileArchiveBundle
from i6_core.util import chunks, MultiPath

from ..cache_manager import cache_file


class RasrFeaturesToHdf(Job):
    __sis_hash_exclude__ = {"out_num_hdfs": 100, "tmp_dir": "/var/tmp"}

    def __init__(
        self,
        feature_caches: typing.Union[MultiPath, typing.List[Path], Path],
        out_num_hdfs: int = 100,
        tmp_dir: typing.Optional[str] = "/var/tmp",
    ):
        self.feature_caches = feature_caches
        self.tmp_dir = tmp_dir

        self.out_hdf_files = [self.output_path(f"data.hdf.{i}", cached=False) for i in range(out_num_hdfs)]
        self.out_num_hdfs = out_num_hdfs
        self.out_single_segment_files = [self.output_path(f"segments.{i}", cached=False) for i in range(out_num_hdfs)]
        self.out_single_seq_lens = [self.output_var(f"seq_lens.{i}") for i in range(out_num_hdfs)]
        self.out_all_seq_lens = self.output_var("all_seq_lens")

        self.rqmt = {"cpu": 1, "mem": 4, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=list(range(self.out_num_hdfs)), parallel=15)
        yield Task("merge_seq_lens", mini_task=True)

    def run(self, *indices: int):
        to_sleep = random.randrange(0, 120)
        logging.info(f"sleeping for {to_sleep}s to avoid thundering herd...")
        time.sleep(to_sleep)

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as bundle_dir:
            if isinstance(self.feature_caches, Path):
                cached_path = cache_file(self.feature_caches)
                cached_bundle = FileArchiveBundle(cached_path)
            else:
                # write paths into temp bundle
                paths = (
                    list(self.feature_caches.hidden_paths.values())
                    if isinstance(self.feature_caches, MultiPath)
                    else self.feature_caches
                )
                paths = [p.get_path() if isinstance(p, Path) else p for p in paths]

                bundle_file_name = os.path.join(bundle_dir, "features.bundle")
                logging.info(f"setting up bundle at {bundle_file_name} from {len(paths)} individual file paths...")

                with open(bundle_file_name, "wt") as bundle_file:
                    bundle_file.writelines([f"{p}\n" for p in paths])

                cached_path = cache_file(bundle_file_name)
                cached_bundle = FileArchiveBundle(cached_path)

            chunked_sequence_list = list(chunks(list(cached_bundle.file_list()), self.out_num_hdfs))

            for i, index in enumerate(indices):
                if i > 0:
                    to_sleep = random.randrange(0, 120)
                    logging.info(f"sleeping for {to_sleep}s to avoid thundering herd...")
                    time.sleep(to_sleep)

                self.process(index, chunked_sequence_list[index], cached_bundle)

    def process(self, index: int, sequences_to_add: typing.List[str], feature_cache: FileArchiveBundle):
        seq_names = []
        seq_lens = {}

        string_dt = h5py.special_dtype(vlen=str)

        logging.info(f"processing chunk {index} with {len(sequences_to_add)} sequences")

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as out_dir:
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

                for file in sequences_to_add:
                    if file.endswith(".attribs"):
                        continue

                    seq_names.append(file)

                    # features
                    times, features = feature_cache.read(file, "feat")
                    seq_lens[file] = len(features)
                    feature_data.create_dataset(seq_names[-1].replace("/", "\\"), data=features)

                out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)

            target = self.out_hdf_files[index].get_path()
            logging.info(f"moving {out_file} to its target {target}")

            shutil.move(out_file, target)

        with open(self.out_single_segment_files[index], "wt") as file:
            file.writelines((f"{seq_name.strip()}\n" for seq_name in seq_names))

        self.out_single_seq_lens[index].set(seq_lens)

    def merge_seq_lens(self):
        self.out_all_seq_lens.set(
            {k: v for seq_lens_map in self.out_single_seq_lens for k, v in seq_lens_map.get().items()},
        )
