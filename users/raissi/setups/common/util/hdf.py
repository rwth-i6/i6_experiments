__all__ = ["RasrFeaturesToHdf", "RasrAlignmentToHDF", "RasrForcedTriphoneAlignmentToHDF"]

import dataclasses
from dataclasses import dataclass
import h5py
import logging
import numpy as np
import os
import random
import shutil
import tempfile
import time
import typing

from sisyphus import Job, Path, Task, tk

from i6_core.lib.rasr_cache import FileArchive
from i6_core.util import MultiPath

from i6_experiments.users.raissi.setups.common.util.cache_manager import cache_file


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


class RasrAlignmentToHDF(Job):
    def __init__(self, alignment_bundle: tk.Path, allophones: tk.Path, state_tying: tk.Path, num_tied_classes: int):
        self.alignment_bundle = alignment_bundle
        self.allophones = allophones
        self.num_tied_classes = num_tied_classes
        self.state_tying = state_tying

        self.out_hdf_file = self.output_path("alignment.hdf")
        self.out_segments = self.output_path("segments")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f = os.path.join(tmp_dir, "data.hdf")
            logging.info(f"processing using temporary file {f}")

            with h5py.File(f, "w") as file:
                self.__run(file)

            shutil.move(f, self.out_hdf_file.get_path())

    def __run(self, out: h5py.File):
        string_dt = h5py.special_dtype(vlen=str)

        with open(self.state_tying, "rt") as st:
            state_tying = {k: int(v) for line in st for k, v in [line.strip().split()[0:2]]}

        cached_alignment_bundle = cache_file(self.alignment_bundle)
        alignment_cache = FileArchiveBundle(cached_alignment_bundle)
        alignment_cache.setAllophones(self.allophones.get_path())

        seq_names = []

        # root
        streams_group = out.create_group("streams")

        # first level
        alignment_group = streams_group.create_group("classes")
        alignment_group.attrs["parser"] = "sparse"
        alignment_group.create_dataset(
            "feature_names",
            data=[b"label_%d" % l for l in range(self.num_tied_classes)],
            dtype=string_dt,
        )

        # second level
        alignment_data = alignment_group.create_group("data")

        for file in alignment_cache.file_list():
            if file.endswith(".attribs"):
                continue

            seq_names.append(file)

            # alignment
            alignment = alignment_cache.read(file, "align")

            alignment_states = [f"{alignment_cache.files[file].allophones[t[1]]}.{t[2]:d}" for t in alignment]
            targets = self.compute_targets(alignment_states=alignment_states, state_tying=state_tying)

            alignment_data.create_dataset(
                seq_names[-1].replace("/", "\\"),
                data=np.array(targets).astype(np.int32),
            )

        out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)

        with open(self.out_segments, "wt") as file:
            file.writelines((f"{seq_name.strip()}\n" for seq_name in seq_names))

    def compute_targets(
        self, alignment_states: typing.List[str], state_tying: typing.Dict[str, int]
    ) -> typing.List[int]:
        targets = [state_tying[allophone] for allophone in alignment_states]
        return targets


@dataclass(eq=True, frozen=True)
class AllophoneState:
    """
    A single, parsed allophone state.
    """

    ctx_l: str
    ctx_r: str
    ph: str
    rest: str

    def __str__(self):
        return f"{self.ph}{{{self.ctx_l}+{self.ctx_r}}}{self.rest}"

    def in_context(
        self, left: typing.Optional["AllophoneState"], right: typing.Optional["AllophoneState"]
    ) -> "AllophoneState":
        if self.ph == "[SILENCE]":
            # Silence does not have context.

            return self

        new_left = left.ph if left is not None and left.ph != "[SILENCE]" else "#"
        new_right = right.ph if right is not None and right.ph != "[SILENCE]" else "#"
        return dataclasses.replace(self, ctx_l=new_left, ctx_r=new_right)

    @classmethod
    def from_alignment_state(cls, state: str) -> "AllophoneState":
        import re

        match = re.match(r"^(.*)\{(.*)\+(.*)}(.*)$", state)
        if match is None:
            raise AttributeError(f"{state} is not an allophone state")

        return cls(ph=match.group(1), ctx_l=match.group(2), ctx_r=match.group(3), rest=match.group(4))


class RasrForcedTriphoneAlignmentToHDF(RasrAlignmentToHDF):
    def compute_targets(
        self, alignment_states: typing.List[str], state_tying: typing.Dict[str, int]
    ) -> typing.List[int]:
        decomposed_alignment = [AllophoneState.from_alignment_state(s) for s in alignment_states]
        in_context = zip((None, *decomposed_alignment[:-1]), decomposed_alignment, (*decomposed_alignment[1:], None))
        forced_triphone_alignment = [str(cur.in_context(prv, nxt)) for prv, cur, nxt in in_context]

        return super().compute_targets(forced_triphone_alignment, state_tying)
