import itertools
import logging
import math
import random
import shutil
import tempfile
import time
from os import path
from typing import List

import h5py
import numpy

from i6_core.lib.rasr_cache import FileArchive
from i6_core.util import chunks
from sisyphus import Job, Path, Task

from ...fh.factored import LabelInfo
from ..cache_manager import cache_file


SHARD_SIZE = 50


class RasrFeatureAndAlignmentWithRandomAllophonesToHDF(Job):
    def __init__(
        self,
        feature_caches: List[Path],
        alignment_caches: List[Path],
        allophones: Path,
        dense_tying: Path,
        cart_tying: Path,
        label_info: LabelInfo,
        randomize: bool = True,
        rng_seed: int = 42,
    ):
        self.feature_caches = feature_caches
        self.alignment_caches = alignment_caches
        self.allophones = allophones
        self.dense_tying = dense_tying
        self.cart_tying = cart_tying
        self.dense_label_info = label_info

        self.rng_seed = rng_seed
        self.is_randomized = randomize

        self.out_hdf_files = [self.output_path("data.hdf.%d" % d, cached=False) for d in range(len(feature_caches))]

        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5 * SHARD_SIZE}

    def tasks(self):
        n = math.ceil(len(self.out_hdf_files) / SHARD_SIZE)
        ch = chunks(list(range(len(self.out_hdf_files))), n)

        yield Task("run", resume="run", rqmt=self.rqmt, args=ch)

    def run(self, *to_process: int):
        logging.info(f"processing files {to_process}")

        for file_index in to_process:
            target_file = self.out_hdf_files[file_index]

            secs = random.randrange(0, 240)
            logging.info(f"sleeping for {secs}s to prevent thundering herd")
            time.sleep(secs)

            with tempfile.TemporaryDirectory() as tmp_dir:
                f = path.join(tmp_dir, "data.hdf")
                logging.info(f"processing file {target_file} using temporary file {f}")

                with h5py.File(f, "w") as file:
                    self.__run(file_index, file)

                shutil.move(f, target_file.get_path())

    def __run(self, file_index: int, out: h5py.File):
        rng = random.Random(self.rng_seed)

        cart_to_dense_dict = self.get_dense_cluster_dict(self.cart_tying, self.dense_tying)
        with open(self.cart_tying, "r") as tying_file:
            cart_tying = {
                k: int(v) for line in tying_file if not line.startswith("#") for k, v in [line.strip().split()[0:2]]
            }
        with open(self.dense_tying, "r") as tying_file:
            dense_tying = {
                k: int(v) for line in tying_file if not line.startswith("#") for k, v in [line.strip().split()[0:2]]
            }

        # allophone and state_tying are generally output of a job, alignment and features might be strings
        feature_path = self.feature_caches[file_index]
        if isinstance(feature_path, Path):
            feature_path = feature_path.get_path()
        feature_path = cache_file(feature_path)
        feature_cache = FileArchive(feature_path)

        alignment_path = self.alignment_caches[min(file_index, len(self.alignment_caches) - 1)]
        if isinstance(alignment_path, Path):
            alignment_path = alignment_path.get_path()
        alignment_path = cache_file(alignment_path)
        alignment_cache = FileArchive(alignment_path)
        alignment_cache.setAllophones(self.allophones.get_path())

        string_dt = h5py.special_dtype(vlen=str)
        seq_names = []

        # num_classes for each label
        n_center_state_classes = self.dense_label_info.get_n_state_classes()
        n_contexts = self.dense_label_info.n_contexts

        # root
        streams_group = out.create_group("streams")

        # first level
        feature_group = streams_group.create_group("features")
        feature_group.attrs["parser"] = "feature_sequence"

        futureLabel_group = streams_group.create_group("futureLabel")
        futureLabel_group.attrs["parser"] = "sparse"
        futureLabel_group.create_dataset(
            "feature_names",
            data=[b"label_%d" % l for l in range(n_contexts)],
            dtype=string_dt,
        )

        centerStateLabel_group = streams_group.create_group("centerState")
        centerStateLabel_group.attrs["parser"] = "sparse"
        centerStateLabel_group.create_dataset(
            "feature_names",
            data=[b"label_%d" % l for l in range(n_center_state_classes)],
            dtype=string_dt,
        )

        pastLabel_group = streams_group.create_group("pastLabel")
        pastLabel_group.attrs["parser"] = "sparse"
        pastLabel_group.create_dataset(
            "feature_names",
            data=[b"label_%d" % l for l in range(n_contexts)],
            dtype=string_dt,
        )

        # second level
        feature_data = feature_group.create_group("data")
        futureLabel_data = futureLabel_group.create_group("data")
        centerstateLabel_data = centerStateLabel_group.create_group("data")
        pastLabel_data = pastLabel_group.create_group("data")

        for file in feature_cache.ft:
            info = feature_cache.ft[file]
            if info.name.endswith(".attribs"):
                continue

            seq_names.append(info.name)

            # features
            times, features = feature_cache.read(file, "feat")
            feature_data.create_dataset(seq_names[-1].replace("/", "\\"), data=features)

            # alignment
            alignment = alignment_cache.read(file, "align")
            aligned_allophones = ["%s.%d" % (alignment_cache.allophones[t[1]], t[2]) for t in alignment]

            # optimize the calculation by grouping
            futureLabel_strings = []
            centerState_strings = []
            pastLabel_strings = []

            if self.is_randomized:
                cart_targets = [cart_tying[allo] for allo in aligned_allophones]
                for k_cart, g in itertools.groupby(cart_targets):
                    segLen = len(list(g))
                    k = rng.choice(cart_to_dense_dict[k_cart])
                    f, c, l = self.get_target_labels_from_dense(k)
                    futureLabel_strings = futureLabel_strings + [f] * segLen
                    centerState_strings = centerState_strings + [c] * segLen
                    pastLabel_strings = pastLabel_strings + [l] * segLen
            else:
                dense_targets = [dense_tying[allo] for allo in aligned_allophones]
                for k, g in itertools.groupby(dense_targets):
                    segLen = len(list(g))
                    f, c, l = self.get_target_labels_from_dense(k)
                    futureLabel_strings = futureLabel_strings + [f] * segLen
                    centerState_strings = centerState_strings + [c] * segLen
                    pastLabel_strings = pastLabel_strings + [l] * segLen

            # initialize last level data
            futureLabel_data.create_dataset(
                seq_names[-1].replace("/", "\\"), data=futureLabel_strings, dtype=numpy.int32
            )
            centerstateLabel_data.create_dataset(
                seq_names[-1].replace("/", "\\"), data=centerState_strings, dtype=numpy.int32
            )
            pastLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=pastLabel_strings, dtype=numpy.int32)

        out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)

    def get_dense_cluster_dict(self, cart_st_path, dense_st_path):
        cart_tying = dict((k, int(v)) for l in open(cart_st_path.get_path()) for k, v in [l.strip().split()[0:2]])
        dense_tying = dict((k, int(v)) for l in open(dense_st_path.get_path()) for k, v in [l.strip().split()[0:2]])

        allophone_clusters = {}
        for k, v in cart_tying.items():
            dense_label = dense_tying[k]
            if v not in allophone_clusters:
                allophone_clusters[v] = []
            allophone_clusters[v].append(dense_label)

        return allophone_clusters

    def get_target_labels_from_dense(self, dense_label: int):
        import numpy as np

        n_contexts = self.dense_label_info.n_contexts

        futureLabel = np.mod(dense_label, n_contexts)
        popFutureLabel = np.floor_divide(dense_label, n_contexts)

        pastLabel = np.mod(popFutureLabel, n_contexts)
        centerState = np.floor_divide(popFutureLabel, n_contexts)

        return futureLabel, centerState, pastLabel
