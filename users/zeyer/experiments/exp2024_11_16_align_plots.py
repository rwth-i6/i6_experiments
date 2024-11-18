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
from typing import Optional, Callable, Tuple, List
import os
import sys
import numpy as np
import pickle
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
    print("seq_tag:", seq_tag)
    print("ref:", get_ref_words())
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


def get_ref_word_boundaries() -> List[Tuple[float, float, str]]:
    out_fn_pickle = out_prefix + seq_tag + "/ref_word_boundaries.pkl"
    if os.path.exists(out_fn_pickle):
        print(f"Already exists: {out_fn_pickle}")
        return pickle.load(open(out_fn_pickle, "rb"))

    ref_words = get_ref_words()

    # adopted from i6_experiments.users.zeyer.experiments.exp2024_09_09_grad_align.CalcAlignmentMetrics
    from returnn.sprint.cache import open_file_archive

    ref_alignment_allophones = Path(
        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
    )
    ref_alignment_sprint_cache = Path(
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
    )
    features_sprint_cache = Path(  # for exact timings
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
    )

    def _cf(fn: Path) -> str:
        return fn.get_path()

    print("Loading ref alignment Sprint cache...")
    ref_align_sprint_cache = open_file_archive(_cf(ref_alignment_sprint_cache))
    print("Loading ref alignment allophones...")
    ref_align_sprint_cache.set_allophones(_cf(ref_alignment_allophones))
    allophones = ref_align_sprint_cache.get_allophones_list()

    print("Loading features Sprint cache...")
    features_sprint_cache = open_file_archive(_cf(features_sprint_cache))

    def _ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    key = seq_tag
    key_ref = "train-other-960/" + "/".join(seq_tag.split("/")[1:])

    print("seq tag:", key)
    feat_times, _ = features_sprint_cache.read(key_ref, typ="feat")
    ref_align = ref_align_sprint_cache.read(key_ref, typ="align")
    assert len(feat_times) == len(ref_align), f"feat len {len(feat_times)} vs ref align len {len(ref_align)}"
    print(f"  start time: {feat_times[0][0]} sec")
    print(f"  end time: {feat_times[-1][1]} sec")

    ref_start_time = ref_align[0][0]
    duration_sec = feat_times[-1][1] - ref_start_time
    sampling_rate = 16_000
    len_samples = round(duration_sec * sampling_rate)  # 16 kHz
    print(f"  num samples: {len_samples} (rounded from {duration_sec * sampling_rate})")
    # RETURNN uses log mel filterbank features, 10ms frame shift, via stft (valid padding)
    window_len = 0.025  # 25 ms
    step_len = 0.010  # 10 ms
    window_num_frames = int(window_len * sampling_rate)
    step_num_frames = int(step_len * sampling_rate)
    len_feat = _ceil_div(len_samples - (window_num_frames - 1), step_num_frames)
    print(f"  num features (10ms): {len_feat} (window {window_num_frames} step {step_num_frames})")
    # ref_alignment_len_factor =
    # len_feat_downsampled = _ceil_div(len_feat, self.ref_alignment_len_factor)
    # print(f"  downsampled num features: {len_feat_downsampled} (factor {self.ref_alignment_len_factor})")

    ref_word_boundaries: List[Tuple[float, float]] = []
    cur_word_start_frame = None
    prev_allophone_idx = None
    ref_words_phones = []
    num_sil_frames_ref = 0
    for t, (t_, allophone_idx, hmm_state_idx, _) in enumerate(ref_align):
        assert t == t_
        if "[SILENCE]" in allophones[allophone_idx]:
            num_sil_frames_ref += 1
            continue
        if cur_word_start_frame is None:
            cur_word_start_frame = t  # new word starts here
            ref_words_phones.append([])
        if prev_allophone_idx != allophone_idx:
            ref_words_phones[-1].append(allophones[allophone_idx])
        if "@f" in allophones[allophone_idx] and (
            t == len(ref_align) - 1 or ref_align[t + 1][1] != allophone_idx or ref_align[t + 1][2] < hmm_state_idx
        ):
            # end of word
            start_time = feat_times[cur_word_start_frame][0] - ref_start_time
            end_time = feat_times[t][1] - ref_start_time
            # take center 10ms of the 25ms window
            # (or not, as we also don't do for the alignment)
            # start_time += (window_len - step_len) / 2
            # end_time -= (window_len - step_len) / 2
            ref_word_boundaries.append((start_time, end_time))
            cur_word_start_frame = None
        prev_allophone_idx = allophone_idx
    assert cur_word_start_frame is None  # word should have ended
    assert len(ref_words_phones) == len(ref_word_boundaries) == len(ref_words)

    # If we would want the phones: " ".join(p[: p.index("{")] for p in phones)
    # But we use the words.
    ref_word_boundaries_ = [
        (float(start_time), float(end_time), word)
        for (start_time, end_time), word in zip(ref_word_boundaries, ref_words)
    ]

    print(f"  num words: {len(ref_word_boundaries)}")
    print(f"  num silence frames: {num_sil_frames_ref}")
    print("save to:", out_fn_pickle)
    pickle.dump(ref_word_boundaries_, open(out_fn_pickle, "wb"))

    return ref_word_boundaries_


def get_ref_words() -> List[str]:
    out_fn_pkl = out_prefix + seq_tag + "/ref_words.pkl"
    if os.path.exists(out_fn_pkl):
        print(f"Already exists: {out_fn_pkl}")
        return pickle.load(open(out_fn_pkl, "rb"))

    from returnn.datasets.audio import OggZipDataset

    dataset = OggZipDataset(
        os.readlink("output/librispeech/dataset/train-clean-100"), targets={"class": "Utf8ByteTargets"}, audio=None
    )
    dataset.init_seq_order(epoch=1, seq_list=[seq_tag])
    dataset.load_seqs(0, 1)
    ref_bytes = dataset.get_data(0, "classes")  # [T, D]
    print(f"ref_bytes.shape: {ref_bytes.shape}")
    print(f"ref_bytes: {ref_bytes!r}")

    ref_str = dataset.targets.get_seq_labels(ref_bytes)
    print(f"ref_str: {ref_str}")
    print(f"ref_str repr: {ref_str!r} type {type(ref_str).__name__}")

    # Split by space.
    # Lowercase all, as Librispeech does not have casing.
    ref_words = [w.lower() for w in ref_str.split()]
    print(f"ref_words: {ref_words}")

    print("save to:", out_fn_pkl)
    pickle.dump(ref_words, open(out_fn_pkl, "wb"))
    return ref_words


def get_word_boundaries_from_hdf_alignment(hdf_name: str) -> List[Tuple[float, float, str]]:
    # TODO...
    return []


def plot_audio_features(*, plotter: Optional[Plotter] = None):
    out_fn_pdf = out_prefix + seq_tag + "/audio_features.pdf"
    audio_features = get_audio_features()
    ref_word_boundaries = get_ref_word_boundaries()

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

    plotter.add_plot("audio", _plot, ref_word_boundaries, rate=100)


def plot_grad_scores(*, plotter: Optional[Plotter] = None):
    out_fn_pdf = out_prefix + seq_tag + "/visualize_grad_scores/" + input_grad_name + "/grads.pdf"

    score_matrix = get_grad_scores()
    S, T = score_matrix.shape  # noqa
    print(f"{input_grad_name}, seq {seq_tag}, shape (SxT) {score_matrix.shape}")
    score_matrix = _log_softmax(np.log(score_matrix), axis=1)  # [S, T]

    # TODO get grad align word boundaries
    get_word_boundaries_from_hdf_alignment(...)

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
        self.plot_titles: List[str] = []
        self.plot_callbacks: List[Callable] = []
        self.plot_word_boundaries: List[Optional[List[Tuple[float, float, str]]]] = []
        self.plot_rates: List[int] = []

        self.fig = None
        self.ax = None

    def add_plot(
        self,
        title: str,
        callback: Callable,
        word_boundaries: Optional[List[Tuple[float, float, str]]] = None,
        *,
        rate: int,
    ):
        self.plot_titles.append(title)
        self.plot_callbacks.append(callback)
        self.plot_word_boundaries.append(word_boundaries)
        self.plot_rates.append(rate)
        self.num_figs += 1

    def make(self):
        self.fig, self.ax = plt.subplots(nrows=self.num_figs, ncols=1, figsize=(20, 5 * self.num_figs))
        if self.num_figs == 1:
            self.ax = [self.ax]

        for i, (title, callback, word_boundaries, rate) in enumerate(
            zip(self.plot_titles, self.plot_callbacks, self.plot_word_boundaries, self.plot_rates)
        ):
            ax = self.ax[i]

            callback(ax)

            ax.set_xlabel("time [sec]")
            ticks = np.arange(0, int(ax.get_xlim()[1] / rate) + 1, 1)
            # ticks = ax.get_xticks() / rate
            # ticks = [round(t, 2) for t in ticks]
            # ticks = [int(t) if t == int(t) else t for t in ticks]
            # ax.set_xticklabels(ticks)
            ax.set_xticks(ticks * rate, ticks)

            if i == self.num_figs - 1:
                ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
                ax.xaxis.set_label_position("bottom")
            elif i == 0:
                ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                ax.xaxis.set_label_position("top")
            else:
                ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

            if word_boundaries:
                for start, end, word in word_boundaries:
                    # ax.axvline(start * rate, color="black", linestyle="--")
                    # ax.axvline(end * rate, color="black", linestyle="--")
                    ax.axvspan(start * rate, end * rate, color="gray", alpha=0.2)
                    ax.text(
                        (start + end) * rate / 2,
                        ax.get_ylim()[1] - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # Add padding
                        word,
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="center",
                        fontsize=12,
                    )

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
