__all__ = ["build_hdf_from_alignment",
           "build_rasr_feature_hdfs"
           "RasrFeaturesToHdf",
           "RasrAlignmentToHDF",
           "RasrForcedTriphoneAlignmentToHDF"]

from sisyphus import gs, Job, Path, Task, tk

from i6_core.corpus import SegmentCorpusJob
import i6_core.features as features
from i6_core.lib.rasr_cache import FileArchive
from i6_core.meta.system import CorpusObject
from i6_core import rasr
from i6_core.returnn import ReturnnDumpHDFJob
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_core.util import MultiPath

import dataclasses
from dataclasses import dataclass
import h5py
from IPython import embed
import logging
import numpy as np
import os
import random
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from i6_experiments.users.raissi.setups.common.util.cache_manager import cache_file
from i6_experiments.common.setups.rasr.util import (
    ReturnnRasrDataInput
)



#### Using ReturnnHDFDump #######

#copied from simon
def build_hdf_from_alignment(
    alignment_cache: tk.Path,
    allophone_file: tk.Path,
    state_tying_file: tk.Path,
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    silence_phone: str = "[SILENCE]",
):
    dataset_config = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": alignment_cache,
                "data_type": "align",
                "allophone_labeling": {
                    "silence_phone": silence_phone,
                    "allophone_file": allophone_file,
                    "state_tying_file": state_tying_file,
                },
            }
        },
    }

    hdf_file = ReturnnDumpHDFJob(
        dataset_config, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    ).out_hdf

    return hdf_file

#copied from simon
def build_rasr_feature_hdfs(
    data_input: ReturnnRasrDataInput,
    feature_name: str,
    feature_extraction_args: Dict[str, Any],

    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    single_hdf: bool = False,
) -> List[tk.Path]:

    hdf_files = []

    if single_hdf or data_input.features is None:
        feature_job = {"mfcc": features.MfccJob, "gt": features.GammatoneJob, "energy": features.EnergyJob,
                       "fb": features.FilterbankJob}[feature_name](
            crp=data_input.crp, **feature_extraction_args
        )
        feature_job.set_keep_value(gs.JOB_DEFAULT_KEEP_VALUE - 20)
        dataset_config = {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": feature_job.out_feature_bundle[feature_name],
                    "data_type": "feat",
                }
            },
        }
        hdf_file = ReturnnDumpHDFJob(
            dataset_config, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
        ).out_hdf
        hdf_files.append(hdf_file)
    else:
        for idx, feature_cache in data_input.features.hidden_paths.items():
            dataset_config = {
                "class": "SprintCacheDataset",
                "data": {
                    "data": {
                        "filename": feature_cache,
                        "data_type": "feat",
                    }
                },
            }

            hdf_file = ReturnnDumpHDFJob(
                dataset_config, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
            ).out_hdf
            hdf_files.append(hdf_file)

    return hdf_files

#### NextGenDataset#####
#From old recipes
class RasrFeaturesToHdf(Job):

    def __init__(self, feature_caches: Union[MultiPath, List[Path]]):
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
        self, alignment_states: List[str], state_tying: Dict[str, int]
    ) -> List[int]:
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
        self, left: Optional["AllophoneState"], right: Optional["AllophoneState"]
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
        self, alignment_states: List[str], state_tying: Dict[str, int]
    ) -> List[int]:
        decomposed_alignment = [AllophoneState.from_alignment_state(s) for s in alignment_states]
        in_context = zip((None, *decomposed_alignment[:-1]), decomposed_alignment, (*decomposed_alignment[1:], None))
        forced_triphone_alignment = [str(cur.in_context(prv, nxt)) for prv, cur, nxt in in_context]

        return super().compute_targets(forced_triphone_alignment, state_tying)


###ToDo------ Needs cleanup
class RasrFeatureAndDeduplicatedPhonemeSequenceToHDF(Job):
    def __init__(self, feature_caches, alignment_bundle, allophones, state_tying):
        self.feature_caches = feature_caches
        self.alignment_bundle = alignment_bundle
        self.state_tying = state_tying
        self.allophones = allophones
        self.hdf_files = [self.output_path("data.hdf.%d" % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        num_classes = 0
        for line in open(self.state_tying.get_path(), "rt"):
            if not line.startswith("#"):
                num_classes = max(num_classes, int(line.strip().split()[1]))

        string_dt = h5py.special_dtype(vlen=str)
        state_tying = dict((k, int(v)) for l in open(self.state_tying.get_path()) for k, v in [l.strip().split()[0:2]])

        feature_cache = FileArchive(self.feature_caches[task_id - 1].get_path())

        alignment_path = self.alignment_bundle.get_path()
        alignment_cache = rasr_cache.open_file_archive(alignment_path)
        allo_align_path = list(alignment_cache.archives.keys())[0]
        alignment_cache.setAllophones(self.allophones)
        allophones = alignment_cache.archives[allo_align_path].allophones

        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), "w")

        # root
        streams_group = out.create_group("streams")

        # first level
        feature_group = streams_group.create_group("features")
        feature_group.attrs["parser"] = "feature_sequence"

        alignment_group = streams_group.create_group("alignment")
        alignment_group.attrs["parser"] = "sparse"
        alignment_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(num_classes + 1)], dtype=string_dt
        )

        # second level
        feature_data = feature_group.create_group("data")
        alignment_data = alignment_group.create_group("data")

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

            targets = []

            alignmentStates = ["%s.%d" % (allophones[t[1]], t[2]) for t in alignment]
            import itertools as it

            for allophone, g in it.groupby(alignmentStates):
                if "SILENCE" not in allophone:
                    targets.append(state_tying[allophone])

            alignment_data.create_dataset(seq_names[-1].replace("/", "\\"), data=targets)

        out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)


###ToDo------ Needs cleanup
class RasrFeatureAndDeduplicatedPhonemeWithWESilenceSequenceToHDF(Job):
    def __init__(self, feature_caches, alignment_bundle, allophones, state_tying):
        self.feature_caches = feature_caches
        self.alignment_bundle = alignment_bundle
        self.state_tying = state_tying
        self.allophones = allophones
        self.hdf_files = [self.output_path("data.hdf.%d" % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        num_classes = 0
        for line in open(self.state_tying.get_path(), "rt"):
            if not line.startswith("#"):
                num_classes = max(num_classes, int(line.strip().split()[1]))

        string_dt = h5py.special_dtype(vlen=str)
        state_tying = dict((k, int(v)) for l in open(self.state_tying.get_path()) for k, v in [l.strip().split()[0:2]])

        feature_cache = FileArchive(self.feature_caches[task_id - 1].get_path())

        alignment_path = self.alignment_bundle.get_path()
        alignment_cache = rasr_cache.open_file_archive(alignment_path)
        allo_align_path = list(alignment_cache.archives.keys())[0]
        alignment_cache.setAllophones(self.allophones)
        allophones = alignment_cache.archives[allo_align_path].allophones

        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), "w")

        # root
        streams_group = out.create_group("streams")

        # first level
        feature_group = streams_group.create_group("features")
        feature_group.attrs["parser"] = "feature_sequence"

        alignment_group = streams_group.create_group("alignment")
        alignment_group.attrs["parser"] = "sparse"
        alignment_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(num_classes + 1)], dtype=string_dt
        )

        # second level
        feature_data = feature_group.create_group("data")
        alignment_data = alignment_group.create_group("data")

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

            targets = [state_tying["[SILENCE]{#+#}@i@f.0"]]

            alignmentStates = ["%s" % (allophones[t[1]]) for t in alignment]
            import itertools as it

            for allophone, g in it.groupby(alignmentStates):
                if "SILENCE" not in allophone:
                    targets.append(state_tying[f"{allophone}.0"] // 3)
                    if "@f" in allophone:
                        targets.append(state_tying["[SILENCE]{#+#}@i@f.0"] // 3)

            alignment_data.create_dataset(seq_names[-1].replace("/", "\\"), data=targets)

        out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)


###ToDo------ Needs cleanup
class RasrFeatureAndAlignmentWithRandomAllophonesToHDF(Job):
    def __init__(self, feature_caches, alignment_caches, allophones, dense_tying, cart_tying, label_info):
        self.feature_caches = feature_caches
        self.alignment_caches = alignment_caches
        self.allophones = allophones
        self.dense_tying = dense_tying
        self.cart_tying = cart_tying
        self.dense_label_info = label_info
        self.hdf_files = [self.output_path("data.hdf.%d" % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        import itertools as it
        import random

        random.seed(42)

        cart_to_dense_dict = self.get_dense_cluster_dict(self.cart_tying, self.dense_tying)
        cart_tying = dict((k, int(v)) for l in open(cart_st_path.get_path()) for k, v in [l.strip().split()[0:2]])

        # allophone and state_tying are generally output of a job, alignment and features might be strings
        feature_path = self.feature_caches[task_id - 1]
        if isinstance(feature_path, tk.Path):
            feature_path = feature_path.get_path()
        feature_cache = FileArchive(feature_path)

        alignment_path = self.alignment_caches[min(task_id - 1, len(self.alignment_caches) - 1)]
        if isinstance(alignment_path, tk.Path):
            alignment_path = alignment_path.get_path()
        alignment_cache = FileArchive(alignment_path)
        alignment_cache.setAllophones(self.allophones.get_path())

        string_dt = h5py.special_dtype(vlen=str)
        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), "w")

        # num_classes for each label
        n_center_state_classes = self.dense_label_info.n_contexts * self.dense_label_info.n_states_per_phone
        n_contexts = self.dense_label_info.n_contexts

        # root
        streams_group = out.create_group("streams")

        # first level
        feature_group = streams_group.create_group("features")
        feature_group.attrs["parser"] = "feature_sequence"

        futureLabel_group = streams_group.create_group("futureLabel")
        futureLabel_group.attrs["parser"] = "sparse"
        futureLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(n_contexts)], dtype=string_dt
        )

        centerStateLabel_group = streams_group.create_group("centerState")
        centerStateLabel_group.attrs["parser"] = "sparse"
        centerStateLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(n_center_state_classes)], dtype=string_dt
        )

        pastLabel_group = streams_group.create_group("pastLabel")
        pastLabel_group.attrs["parser"] = "sparse"
        pastLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(n_contexts)], dtype=string_dt
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
            cart_targets = [cart_tying[allo] for allo in aligned_allophones]

            # optimize the calculation by grouping
            futureLabel_strings = []
            centerState_strings = []
            pastLabel_strings = []
            for k_cart, g in it.groupby(cart_targets):
                segLen = len(list(g))
                k = random.choice(cart_to_dense_dict[k_cart])
                f, c, l = self.get_target_labels_from_dense(k)
                futureLabel_strings = futureLabel_strings + [f] * segLen
                centerState_strings = centerState_strings + [c] * segLen
                pastLabel_strings = pastLabel_strings + [l] * segLen

            # initialize last level data
            futureLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=futureLabel_strings)
            centerstateLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=centerState_strings)
            pastLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=pastLabel_strings)

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

        return allophone_clusters, num_classes

    def get_target_labels_from_dense(self, dense_label):
        import numpy as np

        n_contexts = self.dense_label_info.n_contexts

        futureLabel = np.mod(dense_label, n_contexts)
        popFutureLabel = np.floor_divide(dense_label, n_contexts)

        pastLabel = np.mod(popFutureLabel, n_contexts)
        centerState = np.floor_divide(popFutureLabel, n_contexts)

        return futureLabel, centerState, pastLabel


###ToDo------ Needs cleanup
class RasrFeatureAndAlignmentWithDenseAndCARTStateTyingsToHDF(Job):
    def __init__(self, feature_caches, alignment_caches, allophones, dense_tying, cart_tying, label_info):
        self.feature_caches = feature_caches
        self.alignment_caches = alignment_caches
        self.allophones = allophones
        self.dense_tying = dense_tying
        self.cart_tying = cart_tying
        self.dense_label_info = label_info
        self.hdf_files = [self.output_path("data.hdf.%d" % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        dense_tying, _ = self.get_tying_and_num_classes(self.dense_tying)
        cart_tying, cart_num_classes = self.get_tying_and_num_classes(self.cart_tying)

        # allophone and state_tying are generally output of a job, alignment and features might be strings
        feature_path = self.feature_caches[task_id - 1]
        if isinstance(feature_path, tk.Path):
            feature_path = feature_path.get_path()
        feature_cache = FileArchive(feature_path)

        alignment_path = self.alignment_caches[min(task_id - 1, len(self.alignment_caches) - 1)]
        if isinstance(alignment_path, tk.Path):
            alignment_path = alignment_path.get_path()
        alignment_cache = FileArchive(alignment_path)
        alignment_cache.setAllophones(self.allophones.get_path())

        string_dt = h5py.special_dtype(vlen=str)
        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), "w")

        # num_classes for each label
        n_center_state_classes = self.dense_label_info.n_contexts * self.dense_label_info.n_states_per_phone
        n_contexts = self.dense_label_info.n_contexts

        # root
        streams_group = out.create_group("streams")

        # first level
        feature_group = streams_group.create_group("features")
        feature_group.attrs["parser"] = "feature_sequence"

        cartLabel_group = streams_group.create_group("cartLabel")
        cartLabel_group.attrs["parser"] = "sparse"
        cartLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(cart_num_classes)], dtype=string_dt
        )

        futureLabel_group = streams_group.create_group("futureLabel")
        futureLabel_group.attrs["parser"] = "sparse"
        futureLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(n_contexts)], dtype=string_dt
        )

        centerStateLabel_group = streams_group.create_group("centerState")
        centerStateLabel_group.attrs["parser"] = "sparse"
        centerStateLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(n_center_state_classes)], dtype=string_dt
        )

        pastLabel_group = streams_group.create_group("pastLabel")
        pastLabel_group.attrs["parser"] = "sparse"
        pastLabel_group.create_dataset(
            "feature_names", data=[b"label_%d" % l for l in range(n_contexts)], dtype=string_dt
        )

        # second level
        feature_data = feature_group.create_group("data")
        cartLabel_data = cartLabel_group.create_group("data")
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

            cart_targets = [cart_tying[allo] for allo in aligned_allophones]
            dense_targets = [dense_tying[allo] for allo in aligned_allophones]

            # optimize the calculation by grouping
            futureLabel_strings = []
            centerState_strings = []
            pastLabel_strings = []
            for k, g in it.groupby(dense_targets):
                segLen = len(list(g))
                f, c, l = self.get_target_labels_from_dense(k)
                futureLabel_strings = futureLabel_strings + [f] * segLen
                centerState_strings = centerState_strings + [c] * segLen
                pastLabel_strings = pastLabel_strings + [l] * segLen

            # initialize last level data
            cartLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=cart_targets)
            futureLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=futureLabel_strings)
            centerstateLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=centerState_strings)
            pastLabel_data.create_dataset(seq_names[-1].replace("/", "\\"), data=pastLabel_strings)

        out.create_dataset("seq_names", data=[s.encode() for s in seq_names], dtype=string_dt)

    def get_tying_and_num_classes(self, state_tying_path):
        num_classes = 0
        for line in open(state_tying_path.get_path(), "rt"):
            if not line.startswith("#"):
                num_classes = max(num_classes, int(line.strip().split()[1]))

        state_tying = dict((k, int(v)) for l in open(state_tying_path.get_path()) for k, v in [l.strip().split()[0:2]])

        return state_tying, num_classes

    def get_target_labels_from_dense(self, dense_label):
        import numpy as np

        n_contexts = self.dense_label_info.n_contexts

        futureLabel = np.mod(dense_label, n_contexts)
        popFutureLabel = np.floor_divide(dense_label, n_contexts)

        pastLabel = np.mod(popFutureLabel, n_contexts)
        centerState = np.floor_divide(popFutureLabel, n_contexts)

        return futureLabel, centerState, pastLabel
