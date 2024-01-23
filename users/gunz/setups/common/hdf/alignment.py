__all__ = ["RasrAlignmentToHDF", "RasrForcedTriphoneAlignmentToHDF"]

import dataclasses

import h5py
import logging
import numpy as np
from os import path
import shutil
import tempfile
import typing

from sisyphus import tk, Job, Task

from i6_core.lib.rasr_cache import FileArchiveBundle

from ..analysis.allophone_state import AllophoneState
from ...common.cache_manager import cache_file


class RasrAlignmentToHDF(Job):
    __sis_hash_exclude__ = {"tmp_dir": "/var/tmp", "remap_segment_names": None}

    def __init__(
        self,
        alignment_bundle: tk.Path,
        allophones: tk.Path,
        state_tying: tk.Path,
        num_tied_classes: int,
        tmp_dir: typing.Optional[str] = "/var/tmp",
        remap_segment_names: typing.Optional[typing.Callable[[str], str]] = None,
    ):
        self.alignment_bundle = alignment_bundle
        self.allophones = allophones
        self.num_tied_classes = num_tied_classes
        self.remap_segment_names = remap_segment_names
        self.state_tying = state_tying
        self.tmp_dir = tmp_dir

        self.out_hdf_file = self.output_path("alignment.hdf")
        self.out_segments = self.output_path("segments")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp_dir:
            f = path.join(tmp_dir, "data.hdf")
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

            mapped_name = file if self.remap_segment_names is None else self.remap_segment_names(file)
            seq_names.append(mapped_name)

            # alignment
            alignment = alignment_cache.read(file, "align")

            alignment_states = [f"{alignment_cache.files[file].allophones[t[1]]}.{t[2]:d}" for t in alignment]
            targets = self.compute_targets(alignment_states=alignment_states, state_tying=state_tying)

            alignment_data.create_dataset(
                mapped_name.replace("/", "\\"),
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


class RasrForcedTriphoneAlignmentToHDF(RasrAlignmentToHDF):
    first = True

    def compute_targets(
        self, alignment_states: typing.List[str], state_tying: typing.Dict[str, int]
    ) -> typing.List[int]:
        decomposed_alignment = [AllophoneState.from_alignment_state(s) for s in alignment_states]
        forced_triphone_alignment = []

        for i, a_st in enumerate(decomposed_alignment):
            if a_st.ph == "[SILENCE]":
                forced_triphone_alignment.append(str(a_st))
                continue

            next_left = (st.as_context() for st in decomposed_alignment[i::-1] if st.ph != a_st.ph)
            next_right = (st.as_context() for st in decomposed_alignment[i:] if st.ph != a_st.ph)

            in_ctx = dataclasses.replace(a_st, ctx_l=next(next_left, "#"), ctx_r=next(next_right, "#"))

            forced_triphone_alignment.append(str(in_ctx))

        if self.first:
            self.first = False
            logging.info(f"Example alignment: {forced_triphone_alignment}")

        return super().compute_targets(forced_triphone_alignment, state_tying)
