"""
Generate plots

Actually currently not really for Sisyphus, but standalone script...

To run many of the things here:

Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"

Then: python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_16_align_plots import ... as f; f()"
For example: python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_16_align_plots import all; all()"

Similar as :func:`i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.visualize_grad_scores`.
"""

from __future__ import annotations
from typing import Optional
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


def all():
    plot_grad_scores()


def plot_grad_scores():
    score_matrix_hdf = Path(f"output/exp2024_09_16_grad_align/{input_grad_name}/input_grads.hdf")
    score_matrix_data_dict = load_hdf_data(score_matrix_hdf, num_dims=2)
    basename_tags = {os.path.basename(tag): tag for tag in score_matrix_data_dict.keys()}

    plot_dir = out_prefix + seq_tag + "/visualize_grad_scores/" + input_grad_name
    os.makedirs(plot_dir, exist_ok=True)

    seq_tag_ = seq_tag
    if seq_tag_ not in score_matrix_data_dict:
        if os.path.basename(seq_tag_) in basename_tags:
            seq_tag_ = basename_tags[os.path.basename(seq_tag_)]

    score_matrix = score_matrix_data_dict[seq_tag_]  # [S, T]
    S, T = score_matrix.shape  # noqa
    print(f"{input_grad_name}, seq {seq_tag_}, shape (SxT) {score_matrix.shape}")

    score_matrix = _log_softmax(np.log(score_matrix), axis=1)  # [S, T]

    alias = "log softmax"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
    # score_matrix is [S,T]
    mat_ = ax.matshow(score_matrix, cmap="Blues", aspect="auto")
    ax.tick_params(direction="out", length=20, width=2)
    # ax.set_title(f"{alias} for seq {seq_tag}")
    print(f"{alias} for seq {seq_tag_}")
    ax.set_xlabel("time")
    ax.set_ylabel("labels")
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.gca().xaxis.tick_bottom()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mat_, cax=cax, orientation="vertical")

    plt.tight_layout()
    fn = f"{plot_dir}/grads.pdf"
    print("save to:", fn)
    plt.savefig(fn)


def _log_softmax(x: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
    max_score = np.max(x, axis=axis, keepdims=True)
    x = x - max_score
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def _setup():
    import i6_core.util as util

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    font_size = 22
    matplotlib.rcParams.update(
        {"font.size": font_size, "xtick.labelsize": font_size * 0.8, "ytick.labelsize": font_size * 0.8}
    )


_setup()
