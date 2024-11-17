"""
Generate plots

Actually currently not really for Sisyphus, but standalone script...

To run many of the things here:

Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"

Then: python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_16_align_plots import ... as f; f()"
For example:
    python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_16_align_plots import plot_all as f; f()"

Similar as :func:`i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.visualize_grad_scores`.
"""

from __future__ import annotations
from typing import Optional, Callable
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

from sisyphus import Path
from i6_experiments.users.schmitt.hdf import load_hdf_data

# See i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.visualize_grad_scores

# seq_list = Path(
#     "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
# )
# seq_list = open(seq_list.get_path()).read().splitlines()
seq_tag = "train-clean-100/103-1240-0000/103-1240-0000"

input_grad_name = "ctc-grad-align/base"

out_prefix = "output/exp2024_11_16_grad_align/"


def plot_all():
    plotter = Plotter(out_filename=out_prefix + seq_tag + "/combined.pdf")
    plot_audio_features(plotter=plotter)
    plot_grad_scores(plotter=plotter)
    plotter.make()


def get_audio_features():
    out_fn_npz = out_prefix + seq_tag + "/audio_features.npz"
    if os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        audio_features = np.load(out_fn_npz)["audio_features"]

    else:
        from returnn.datasets.audio import OggZipDataset

        dataset = OggZipDataset(
            os.readlink("output/librispeech/dataset/train-clean-100"),
            targets=None,
            audio={"features": "log_mel_filterbank", "num_feature_filters": 120},
            # audio={"features": "mfcc", "num_feature_filters": 80},
        )
        dataset.init_seq_order(epoch=1, seq_list=[seq_tag])
        dataset.load_seqs(0, 1)
        audio_features = dataset.get_data(0, "data")  # [T, D]
        print(f"audio_features.shape: {audio_features.shape}")

        print("save to:", out_fn_npz)
        np.savez(out_fn_npz, audio_features=audio_features)
    return audio_features


def plot_audio_features(*, plotter: Optional[Plotter] = None):
    out_fn_pdf = out_prefix + seq_tag + "/audio_features.pdf"
    audio_features = get_audio_features()

    if not plotter:
        plotter = Plotter(plot_at_del=True, out_filename=out_fn_pdf)

    def _plot(ax):
        # audio_features is [T,D]
        mat_ = ax.matshow(audio_features.T, cmap="Blues", aspect="auto")
        ax.tick_params(direction="out", length=20, width=2)
        # ax.set_title(f"{alias} for seq {seq_tag}")
        print(f"for seq {seq_tag}")

        ax.set_ylabel("feature")
        ax.set_ylim(ax.get_ylim()[::-1])
        # plt.gca().xaxis.tick_bottom()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plotter.fig.colorbar(mat_, cax=cax, orientation="vertical")

    plotter.add_plot("audio", _plot, rate=100)


def get_grad_scores():
    out_fn_npz = out_prefix + seq_tag + "/visualize_grad_scores/" + input_grad_name + "/grads.npz"

    if os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        data = np.load(out_fn_npz)
        score_matrix = data["score_matrix"]

    else:
        score_matrix_hdf = Path(f"output/exp2024_09_16_grad_align/{input_grad_name}/input_grads.hdf")
        score_matrix_data_dict = load_hdf_data(score_matrix_hdf, num_dims=2)
        basename_tags = {os.path.basename(tag): tag for tag in score_matrix_data_dict.keys()}

        seq_tag_ = seq_tag
        if seq_tag_ not in score_matrix_data_dict:
            if os.path.basename(seq_tag_) in basename_tags:
                seq_tag_ = basename_tags[os.path.basename(seq_tag_)]

        score_matrix = score_matrix_data_dict[seq_tag_]  # [S, T]
        print(f"load {score_matrix_hdf}: {seq_tag_}, shape {score_matrix.shape}")
        print(f"save to:", out_fn_npz)
        np.savez(out_fn_npz, seq_tag=seq_tag_, score_matrix=score_matrix)
    return score_matrix


def plot_grad_scores(*, plotter: Optional[Plotter] = None):
    out_fn_pdf = out_prefix + seq_tag + "/visualize_grad_scores/" + input_grad_name + "/grads.pdf"

    score_matrix = get_grad_scores()
    S, T = score_matrix.shape  # noqa
    print(f"{input_grad_name}, seq {seq_tag}, shape (SxT) {score_matrix.shape}")

    score_matrix = _log_softmax(np.log(score_matrix), axis=1)  # [S, T]

    if not plotter:
        plotter = Plotter(plot_at_del=True, out_filename=out_fn_pdf)

    def _plot(ax):
        alias = "log softmax"
        # score_matrix is [S,T]
        mat_ = ax.matshow(score_matrix, cmap="Blues", aspect="auto")
        ax.tick_params(direction="out", length=20, width=2)
        # ax.set_title(f"{alias} for seq {seq_tag}")
        print(f"{alias} for seq {seq_tag}")
        ax.set_ylabel("labels")
        ax.set_ylim(ax.get_ylim()[::-1])
        # plt.gca().xaxis.tick_bottom()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plotter.fig.colorbar(mat_, cax=cax, orientation="vertical")

    plotter.add_plot("grad", _plot, rate=100)


def _log_softmax(x: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
    max_score = np.max(x, axis=axis, keepdims=True)
    x = x - max_score
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


class Plotter:
    def __init__(self, *, plot_at_del: bool = False, out_filename: str):
        self.plot_at_del = plot_at_del
        assert out_filename.endswith(".pdf")
        self.out_filename = out_filename

        self.num_figs = 0
        self.plot_titles = []
        self.plot_callbacks = []
        self.plot_rates = []

        self.fig = None
        self.ax = None

    def add_plot(self, title: str, callback: Callable, *, rate: int):
        self.plot_titles.append(title)
        self.plot_callbacks.append(callback)
        self.plot_rates.append(rate)
        self.num_figs += 1

    def make(self):
        self.fig, self.ax = plt.subplots(nrows=self.num_figs, ncols=1, figsize=(20, 5 * self.num_figs))
        if self.num_figs == 1:
            self.ax = [self.ax]

        for i, (title, callback, rate) in enumerate(zip(self.plot_titles, self.plot_callbacks, self.plot_rates)):
            ax = self.ax[i]

            callback(ax)

            ax.set_xlabel("time (sec)")
            ticks = ax.get_xticks() / rate
            ax.set_xticklabels(ticks)

            if i == self.num_figs - 1:
                ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
                ax.xaxis.set_label_position("bottom")
            elif i == 0:
                ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                ax.xaxis.set_label_position("top")
            else:
                ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

            if self.num_figs > 1:
                # ax.set_title(title, fontweight="bold", x=0, y=1)
                ax.set_title(title)
                # ax.set_title(title, x=1.025, y=-0.48, fontsize=18, fontweight="bold")

        # plt.gca().xaxis.tick_bottom()
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.out_filename), exist_ok=True)
        print("save to:", self.out_filename)
        plt.savefig(self.out_filename)

    def __del__(self):
        if self.plot_at_del:
            self.make()


def _setup():
    import i6_core.util as util

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    font_size = 22
    matplotlib.rcParams.update(
        {"font.size": font_size, "xtick.labelsize": font_size * 0.8, "ytick.labelsize": font_size * 0.8}
    )


_setup()
