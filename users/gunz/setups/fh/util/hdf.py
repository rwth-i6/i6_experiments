import copy
from enum import Enum
import h5py
import logging
import numpy as np
import os
from os import path
import random
import shutil
import tempfile
import time
import typing

from sisyphus import tk, Job, Task

from i6_core.lib.rasr_cache import FileArchive, FileArchiveBundle
from i6_core.util import MultiPath

from ...common.cache_manager import cache_file


class UnderflowBehavior(Enum):
    repeat = "repeat"
    zero_pad = "zero-pad"


class RasrFeatureAndAlignmentToHDF(Job):
    def __init__(
        self,
        feature_caches: typing.Union[MultiPath, typing.List[tk.Path]],
        alignment_bundle: tk.Path,
        allophones: tk.Path,
        state_tying: tk.Path,
        num_tied_classes: int,
        downsampling_factor: int = 1,
        feature_length_underflow_behavior: UnderflowBehavior = UnderflowBehavior.zero_pad,
        random_sleep_duration_secs: int = 60,
        parallel: int = 10,
    ):
        self.alignment_bundle = alignment_bundle
        self.allophones = allophones
        self.downsampling_factor = downsampling_factor
        self.feature_caches = (
            feature_caches if isinstance(feature_caches, list) else list(feature_caches.hidden_paths.values())
        )
        self.feature_length_underflow_behavior = feature_length_underflow_behavior
        self.num_tied_classes = num_tied_classes
        self.parallel = parallel
        self.random_sleep_duration_secs = random_sleep_duration_secs
        self.state_tying = state_tying

        self.out_hdf_files = [self.output_path(f"data.hdf.{i}") for i in range(len(self.feature_caches))]

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task(
            "run",
            resume="run",
            rqmt=self.rqmt,
            args=list(range(len(self.out_hdf_files))),
            parallel=self.parallel,
        )

    def run(self, file_index: int):
        sleep_dur = random.randrange(0, self.random_sleep_duration_secs)

        logging.info(f"processing file {file_index}")
        logging.info(f"sleeping for {sleep_dur}s before commencing processing to avoid thundering herd")

        time.sleep(sleep_dur)

        target_file = self.out_hdf_files[file_index]

        with tempfile.TemporaryDirectory() as tmp_dir:
            f = path.join(tmp_dir, "data.hdf")
            logging.info(f"processing file {target_file} using temporary file {f}")

            with h5py.File(f, "w") as file:
                self.__run(file_index, file)

            shutil.move(f, target_file.get_path())

    def __run(self, file_index: int, out: h5py.File):
        string_dt = h5py.special_dtype(vlen=str)

        with open(self.state_tying, "rt") as st:
            state_tying = {k: int(v) for line in st for k, v in [line.strip().split()[0:2]]}

        feature_file = cache_file(self.feature_caches[file_index])
        feature_cache = FileArchive(feature_file)

        cached_alignment_bundle = cache_file(self.alignment_bundle)
        alignment_cache = FileArchiveBundle(cached_alignment_bundle)
        alignment_cache.setAllophones(self.allophones.get_path())

        seq_names = []

        # root
        streams_group = out.create_group("streams")

        # first level
        feature_group = streams_group.create_group("features")
        feature_group.attrs["parser"] = "feature_sequence"

        alignment_group = streams_group.create_group("classes")
        alignment_group.attrs["parser"] = "sparse"
        alignment_group.create_dataset(
            "feature_names",
            data=[b"label_%d" % l for l in range(self.num_tied_classes)],
            dtype=string_dt,
        )

        # second level
        feature_data = feature_group.create_group("data")
        alignment_data = alignment_group.create_group("data")

        for file in feature_cache.ft:
            info = feature_cache.ft[file]
            if info.name.endswith(".attribs"):
                continue

            seq_names.append(info.name)

            # alignment
            alignment = alignment_cache.read(file, "align")

            alignment_states = [f"{alignment_cache.files[file].allophones[t[1]]}.{t[2]:d}" for t in alignment]
            targets = [state_tying[allophone] for allophone in alignment_states]

            features_min_len = (len(targets) - 1) * self.downsampling_factor
            features_exact_len = len(targets) * self.downsampling_factor
            features_max_len = (len(targets) + 1) * self.downsampling_factor

            alignment_data.create_dataset(
                seq_names[-1].replace("/", "\\"),
                data=np.array(targets).astype(np.int32),
            )

            # features
            times, features = feature_cache.read(file, "feat")

            # ensure features have proper length
            assert (
                len(features) > features_min_len
            ), f"downsampling factor mismatch, difference not a rounding error: {len(targets)}t * {self.downsampling_factor} >> {len(features)}f"
            assert (
                len(features) < features_max_len
            ), f"downsampling factor mismatch, difference not a rounding error: {len(targets)}t * {self.downsampling_factor} << {len(features)}f"

            if len(features) < features_exact_len:
                diff = features_exact_len - len(features)
                padding_element = (
                    np.zeros(np.shape(features[0]))
                    if self.feature_length_underflow_behavior == UnderflowBehavior.zero_pad
                    else copy.deepcopy(features[-1])
                )
                features = features + (diff * [padding_element])
            elif len(features) > features_exact_len:
                features = features[:features_exact_len]

            assert len(features) == features_exact_len

            feature_data.create_dataset(
                seq_names[-1].replace("/", "\\"),
                data=np.array(features).astype(np.float32),
            )

        out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)
