__all__ = ["AlignmentStatisticsJob", "AlignmentVisualisationJob"]

import os
import shutil
import subprocess
from typing import Optional, List

from recipe.i6_core.util import create_executable
from recipe.i6_experiments.users.berger.systems.dataclasses import AlignmentData

from sisyphus import *

import recipe.i6_experiments.users.gruev.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)


class AlignmentStatisticsJob(Job):
    def __init__(
        self, alignment, num_labels, blank_idx=0, seq_list_filter_file=None, returnn_python_exe=None, returnn_root=None,
    ):

        self.returnn_python_exe = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root = returnn_root if returnn_root is not None else gs.RETURNN_ROOT

        self.alignment = alignment
        self.seq_list_filter_file = seq_list_filter_file
        self.num_labels = num_labels
        self.blank_idx = blank_idx

        self.out_statistics = self.output_path("statistics.txt")
        self.out_labels_hist = self.output_path("labels_histogram.pdf")
        self.out_mean_label_seg_lens = self.output_path("mean_label_seg_lens.json")
        # self.out_mean_label_seg_lens_var = self.output_var("mean_label_seg_lens_var.json")
        self.out_mean_label_seg_len = self.output_path("mean_label_seg_len.txt")
        # self.out_mean_label_seg_len_var = self.output_var("mean_label_seg_len_var.txt")
        self.out_mean_label_seq_len = self.output_path("mean_label_seq_len.txt")
        # self.out_mean_label_seq_len_var = self.output_var("mean_label_seq_len_var.txt")
        self.out_90_quantile_var = self.output_var("quantile_90")
        self.out_95_quantile_var = self.output_var("quantile_95")
        self.out_99_quantile_var = self.output_var("quantile_99")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 2})

    def run(self):
        command = [
            self.returnn_python_exe.get_path(),
            os.path.join(tools_dir, "segment_statistics.py"),
            self.alignment.get_path(),
            "--num-labels",
            str(self.num_labels),
            "--blank-idx",
            str(self.blank_idx),
            "--returnn-root",
            self.returnn_root.get_path(),
        ]

        if self.seq_list_filter_file:
            command += ["--seq-list-filter-file", str(self.seq_list_filter_file)]

        create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])

        # with open("mean_label_seg_lens.json", "r") as f:
        #     mean_label_seg_lens = json.load(f)
        #     mean_label_seg_lens = [
        #         mean_label_seg_lens[str(idx)] for idx in range(self.num_labels)
        #     ]
        #     self.out_mean_label_seg_lens_var.set(mean_label_seg_lens)
        #
        # with open("mean_label_seg_len.txt", "r") as f:
        #     self.out_mean_label_seg_len_var.set(float(f.read()))
        #
        # with open("mean_label_seq_len.txt", "r") as f:
        #     self.out_mean_label_seq_len_var.set(float(f.read()))

        # Set quantiles
        with open("quantile_90", "r") as f:
            self.out_90_quantile_var.set(int(float(f.read())))
        with open("quantile_95", "r") as f:
            self.out_95_quantile_var.set(int(float(f.read())))
        with open("quantile_99", "r") as f:
            self.out_99_quantile_var.set(int(float(f.read())))

        shutil.move("statistics.txt", self.out_statistics.get_path())
        shutil.move("labels_histogram.pdf", self.out_labels_hist.get_path())
        shutil.move("mean_label_seg_lens.json", self.out_mean_label_seg_lens.get_path())
        shutil.move("mean_label_seg_len.txt", self.out_mean_label_seg_len.get_path())
        shutil.move("mean_label_seq_len.txt", self.out_mean_label_seq_len.get_path())


class AlignmentVisualisationJob(Job):
    def __init__(
        self,
        alignment: AlignmentData,
        min_segment_length: float = 0.0,
        max_segment_length: float = 30.0,
        num_segments: int = 10,
        segment_whitelist: Optional[List[str]] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):

        self.alignment = alignment

        self.returnn_python_exe = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root = returnn_root if returnn_root is not None else gs.RETURNN_ROOT

        # The following are passed to the alignment visualisation script
        self.alignment_hdf = alignment.get_hdf(returnn_python_exe, returnn_root)
        self.allophone_file = alignment.allophone_file
        self.state_tying_file = alignment.state_tying_file
        self.silence_phone = alignment.silence_phone

        # Optional and by default fixed
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.num_segments = num_segments
        self.segment_whitelist = segment_whitelist

        # For registration purposes
        self.out_path = self.output_path(os.getcwd())

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 2})

    def run(self):
        command = [
            self.returnn_python_exe.get_path(),
            os.path.join(tools_dir, "segment_visualisation.py"),
            self.alignment_hdf.get_path(),
            self.allophone_file.get_path(),
            self.state_tying_file.get_path(),
            self.silence_phone,
            self.returnn_root.get_path(),
            "--min_segment_length",
            str(self.min_segment_length),
            "--max_segment_length",
            str(self.max_segment_length),
            "--num_segments",
            str(self.num_segments),
        ]

        if self.segment_whitelist:
            command += ["--segment_whitelist", str(self.segment_whitelist)]

        create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])
