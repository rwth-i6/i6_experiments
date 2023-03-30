from sisyphus import *
from sisyphus import tools

import subprocess
import os
import shutil
import numpy as np
import copy

from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.default_tools import (
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
)
from i6_core import util
from i6_core.returnn.config import ReturnnConfig


class DumpAttentionWeightsJob(Job):
    def __init__(
        self,
        returnn_config,
        rasr_config,
        rasr_nn_trainer_exe,
        seq_tag,
        hdf_targets,
        blank_idx,
        label_name,
        model_type,
        returnn_python_exe=None,
        returnn_root=None,
    ):

        self.model_type = model_type
        self.label_name = label_name
        self.blank_idx = blank_idx
        self.rasr_nn_trainer_exe = rasr_nn_trainer_exe
        self.rasr_config = rasr_config
        self.hdf_targets = hdf_targets
        self.seq_tag = seq_tag
        self.returnn_config = returnn_config
        self.returnn_python_exe = (
            returnn_python_exe if returnn_python_exe else RETURNN_CPU_EXE
        )
        self.returnn_root = returnn_root if returnn_root else RETURNN_ROOT

        self.out_data = self.output_path("out_data.npz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 6, "time": 1})

    def run(self):
        self.returnn_config.write("returnn.config")

        with open("seq_file", "w+") as f:
            f.write(self.seq_tag + "\n")

        tools_dir = os.path.join(RETURNN_ROOT, "tools")

        command = [
            self.returnn_python_exe,
            os.path.join(tools_dir, "dump_attention_weights.py"),
            "returnn.config",
            "--rasr_nn_trainer_exe",
            self.rasr_nn_trainer_exe.get_path(),
            "--segment_file",
            "seq_file",
            "--rasr_config_path",
            self.rasr_config.get_path(),
            "--hdf_targets",
            self.hdf_targets.get_path(),
            "--blank_idx",
            str(self.blank_idx),
            "--label_name",
            self.label_name,
            "--model_type",
            self.model_type,
            "--returnn_root",
            self.returnn_root,
        ]

        util.create_executable("rnn.sh", command)
        subprocess.check_call(["./rnn.sh"])

        shutil.move("data.npz", self.out_data)


class PlotAttentionWeightsJob(Job):
    def __init__(self, data_path, blank_idx, json_vocab_path, time_red, seq_tag):
        self.data_path = data_path
        self.blank_idx = blank_idx
        self.time_red = time_red
        self.seq_tag = seq_tag
        self.json_vocab_path = json_vocab_path
        self.out_plot = self.output_path("out_plot.png")
        self.out_plot_pdf = self.output_path("out_plot.pdf")

    def tasks(self):
        yield Task(
            "run", rqmt={"cpu": 1, "mem": 2, "time": 1, "gpu": 0}, mini_task=True
        )

    def run(self):
        # load hmm alignment with the rasr archiver and parse the console output
        hmm_align = subprocess.check_output(
            [
                "/u/schmitt/experiments/transducer/config/sprint-executables/archiver",
                "--mode",
                "show",
                "--type",
                "align",
                "--allophone-file",
                "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones",
                "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.8ApwUvz5GMt7/output/alignment.cache.bundle",
                self.seq_tag,
            ]
        )
        hmm_align = hmm_align.splitlines()
        hmm_align = [row.decode("utf-8").strip() for row in hmm_align]
        hmm_align = [row for row in hmm_align if row.startswith("time")]
        hmm_align_states = [row.split("\t")[-1] for row in hmm_align]
        hmm_align = [row.split("\t")[5].split("{")[0] for row in hmm_align]
        hmm_align = [phon if phon != "[SILENCE]" else "[S]" for phon in hmm_align]

        hmm_align_borders = [
            i + 1
            for i, x in enumerate(hmm_align)
            if i < len(hmm_align) - 1
            and (
                hmm_align[i] != hmm_align[i + 1]
                or hmm_align_states[i] > hmm_align_states[i + 1]
            )
        ]
        if hmm_align_borders[-1] != len(hmm_align):
            hmm_align_borders += [len(hmm_align)]
        hmm_align_major = [
            hmm_align[i - 1]
            for i in range(1, len(hmm_align) + 1)
            if i in hmm_align_borders
        ]
        if hmm_align_borders[0] != 0:
            hmm_align_borders = [0] + hmm_align_borders
        hmm_align_borders_center = [
            (i + j) / 2 for i, j in zip(hmm_align_borders[:-1], hmm_align_borders[1:])
        ]

        with open(self.json_vocab_path.get_path(), "r") as f:
            label_to_idx = ast.literal_eval(f.read())
            idx_to_label = {int(v): k for k, v in label_to_idx.items()}
        data_file = np.load(self.data_path.get_path())
        weights = data_file["weights"][:, 0, :]
        align = data_file["labels"]
        seg_starts = None
        seg_lens = None
        if self.blank_idx is not None:
            seg_starts = data_file["seg_starts"]
            seg_lens = data_file["seg_lens"]
            label_idxs = align[align != self.blank_idx]
            labels = [idx_to_label[idx] for idx in label_idxs]
            # label_positions = np.where(align != self.blank_idx)[0]
            zeros = np.zeros(
                (
                    weights.shape[0],
                    (int(np.ceil(len(hmm_align) / self.time_red))) - weights.shape[1],
                )
            )
            weights = np.concatenate([weights, zeros], axis=1)
            weights = np.array(
                [np.roll(row, start) for start, row in zip(seg_starts, weights)]
            )
        else:
            labels = [idx_to_label[idx] for idx in align]

        last_label_rep = len(hmm_align) % self.time_red
        weights = np.concatenate(
            [
                np.repeat(weights[:, :-1], self.time_red, axis=-1),
                np.repeat(weights[:, -1:], last_label_rep, axis=-1),
            ],
            axis=-1,
        )

        fig, ax = plt.subplots(figsize=(45, 12), constrained_layout=True)
        ax.matshow(weights, cmap=plt.cm.get_cmap("Blues"), aspect="auto")
        # time_ax = ax.twiny()

        # for every label, there is a "row" of size 1
        # we want to have a major tick at every row border, i.e. the first is at 1, then 2, then 3 etc
        yticks = [i + 1 for i in range(len(labels))]
        # we want to set the labels between these major ticks, e.g. (i+j)/2
        yticks_minor = [
            ((i + j) / 2) for i, j in zip(([0] + yticks)[:-1], ([0] + yticks)[1:])
        ]
        # set corresponding ticks and labels
        ax.set_yticks([tick - 0.5 for tick in yticks])
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.set_yticks([tick - 0.5 for tick in yticks_minor], minor=True)
        ax.set_yticklabels(labels, fontsize=40, minor=True)
        # disable minor ticks and make major ticks longer
        ax.tick_params(axis="y", which="minor", length=0)
        ax.tick_params(axis="y", which="major", length=10)
        # solve the plt bug that cuts off the first and last voxel in matshow
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_ylabel("Output Labels", fontsize=50)
        for idx in yticks:
            ax.axhline(y=idx - 0.5, xmin=0, xmax=1, color="k", linewidth=0.5)

        xticks = list(hmm_align_borders)[1:]
        xticks_minor = hmm_align_borders_center
        ax.set_xticks([tick - 0.5 for tick in xticks])
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.set_xticks([tick - 0.5 for tick in xticks_minor], minor=True)
        ax.set_xticklabels(hmm_align_major, minor=True, rotation=90)
        ax.tick_params(axis="x", which="minor", length=0, labelsize=40)
        ax.tick_params(axis="x", which="major", length=10, width=0.5)
        ax.set_xlabel("HMM Alignment", fontsize=50, labelpad=20)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        for idx in xticks:
            ax.axvline(
                x=idx - 0.5, ymin=0, ymax=1, color="k", linestyle="--", linewidth=0.5
            )

        num_labels = len(labels)
        if seg_starts is not None:
            for i, (start, length) in enumerate(zip(seg_starts, seg_lens)):
                ax.axvline(
                    x=start * self.time_red - 0.5,
                    ymin=(num_labels - i) / num_labels,
                    ymax=(num_labels - i - 1) / num_labels,
                    color="r",
                )
                ax.axvline(
                    x=min((start + length) * self.time_red, xticks[-1]) - 0.5,
                    ymin=(num_labels - i) / num_labels,
                    ymax=(num_labels - i - 1) / num_labels,
                    color="r",
                )

        # time_ax = ax.twiny()
        # time_ticks = [x - .5 for x in range(0, len(align), 10)]
        # time_labels = [x for x in range(0, len(align), 10)]
        # time_ax.set_xlabel("Input Time Frames", fontsize=20)
        # time_ax.xaxis.tick_bottom()
        # time_ax.xaxis.set_label_position('bottom')
        # # time_ax.set_xlim(ax.get_xlim())
        # time_ax.set_xticks(time_ticks)
        # time_ax.set_xticklabels(time_labels, fontsize=5)

        plt.savefig(self.out_plot.get_path())
        plt.savefig(self.out_plot_pdf.get_path())


class DumpReturnnLayerJob(Job):
    def __init__(
        self,
        returnn_config: ReturnnConfig,
        *,
        rqmt=None,
        returnn_python_exe=None,
        returnn_root=None
    ):
        super().__init__()

        returnn_config = copy.deepcopy(returnn_config)
        returnn_config.config = copy.deepcopy(returnn_config.config)
        returnn_config.config["task"] = "eval"

        self.returnn_config = returnn_config
        self.rqmt = rqmt
        self.returnn_python_exe = util.get_returnn_python_exe(returnn_python_exe)
        self.returnn_root = util.get_returnn_root(returnn_root)

        net_dict = returnn_config.config["network"]
        eval_datasets = {}
        for name in ["dev", "eval"]:
            if returnn_config.config.get(name):
                eval_datasets[name] = returnn_config.config[name]
        if returnn_config.config.get("eval_datasets"):
            eval_datasets.update(returnn_config.config["eval_datasets"])

        self.out_files = {}
        assert isinstance(net_dict, dict)
        for layer_name, layer_dict in net_dict.items():
            if not isinstance(layer_dict, dict):
                continue
            if layer_dict.get("class") != "hdf_dump":
                continue
            filename = layer_dict["filename"]
            assert isinstance(filename, str) and "/" not in filename
            if layer_dict.get("dump_per_run", False):
                out_files_per_run = {}
                for dataset_name in eval_datasets:
                    assert isinstance(dataset_name, str)
                    run_opts = {"dataset_name": dataset_name}  # ignore epoch for now
                    layer_dict["filename"] = "../output/" + filename
                    filename_ = filename.format_map(run_opts)
                    filename_ = self.output_path(filename_)
                    out_files_per_run[dataset_name] = filename_
                self.out_files[layer_name] = out_files_per_run
            else:
                filename = self.output_path(filename)
                layer_dict["filename"] = filename
                self.out_files[layer_name] = filename

    def run(self):
        self.returnn_config.write("returnn.config")
        subprocess.check_call(
            [
                self.returnn_python_exe.get_path(),
                self.returnn_root.get_path() + "/rnn.py",
                "returnn.config",
            ]
        )
        for out in self.out_files.values():
            assert os.path.exists(out.get_path())

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, parsed_args):
        return tools.sis_hash(parsed_args["returnn_config"])
