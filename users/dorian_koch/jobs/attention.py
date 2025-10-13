import copy

from sisyphus import *

from i6_core.util import MultiOutputPath

import subprocess
import tempfile
import shutil
import os
import json
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import ast
import numpy as np
import h5py
from typing import Optional, Any, Dict, Tuple, List, Union
import scipy

import i6_experiments.users.schmitt.tools as tools_mod

tools_dir = os.path.dirname(tools_mod.__file__)

from i6_experiments.users.schmitt import hdf


class PlotAttentionWeightsJobV2(Job):
    font_size_small = 16
    font_size_medium = 20
    font_size_large = 32

    pt_to_inches = 1 / 72.27

    def __init__(
        self,
        att_weight_hdf: Union[Path, List[Path]],
        targets_hdf: Path,
        seg_starts_hdf: Optional[Path],
        seg_lens_hdf: Optional[Path],
        center_positions_hdf: Optional[Path],
        target_blank_idx: Optional[int],
        ref_alignment_blank_idx: int,
        ref_alignment_hdf: Path,
        json_vocab_path: Path,
        ctc_alignment_hdf: Optional[Path] = None,
        segment_whitelist: Optional[List[str]] = None,
        ref_alignment_json_vocab_path: Optional[Path] = None,
        plot_w_cog: bool = False,
        titles: Optional[List[str]] = None,
        vmin: Optional[Union[Dict, int, float]] = 0.0,
        vmax: Optional[Union[Dict, int, float]] = 1.0,
        scale: float = 1.0,
        ref_alignment_is_positional_alignment: Optional[bool] = None,
    ):
        assert target_blank_idx is None or (seg_lens_hdf is not None and seg_starts_hdf is not None)
        self.seg_lens_hdf = seg_lens_hdf
        self.seg_starts_hdf = seg_starts_hdf
        self.center_positions_hdf = center_positions_hdf
        self.targets_hdf = targets_hdf
        self.titles = titles
        self.vmin = vmin
        self.vmax = vmax
        self.scale = scale
        self.ref_alignment_is_positional_alignment = ref_alignment_is_positional_alignment

        if isinstance(att_weight_hdf, list):
            self.att_weight_hdf = att_weight_hdf[0]
            self.other_att_weights = att_weight_hdf[1:]
        else:
            self.att_weight_hdf = att_weight_hdf
            self.other_att_weights = None
        self.target_blank_idx = target_blank_idx
        self.ref_alignment_blank_idx = ref_alignment_blank_idx
        self.ref_alignment_hdf = ref_alignment_hdf
        self.json_vocab_path = json_vocab_path
        self.ctc_alignment_hdf = ctc_alignment_hdf
        self.segment_whitelist = segment_whitelist  # if segment_whitelist is not None else []

        if ref_alignment_json_vocab_path is None:
            self.ref_alignment_json_vocab_path = json_vocab_path
        else:
            self.ref_alignment_json_vocab_path = ref_alignment_json_vocab_path

        self.plot_w_color_gradient = False

        self.out_plot_dir = self.output_path("plots", True)
        self.out_plot_w_ctc_dir = self.output_path("plots_w_ctc", True)

        self.plot_w_cog = plot_w_cog
        if plot_w_cog:
            self.out_plot_w_cog_dir = self.output_path("plots_w_cog", True)
        self.out_plot_path = MultiOutputPath(self, "plots/plots.$(TASK)", self.out_plot_dir)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def load_data(self) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        Load data from the hdf files
        """
        if self.att_weight_hdf.get_path().split(".")[-1] == "hdf":
            att_weights_dict = hdf.load_hdf_data(self.att_weight_hdf, num_dims=2, segment_list=self.segment_whitelist)
            targets_dict = hdf.load_hdf_data(self.targets_hdf, segment_list=self.segment_whitelist)
        else:
            assert self.att_weight_hdf.get_path().split(".")[-1] == "npy"

            att_weights_dict = {}
            targets_dict = {}

            numpy_object = np.load(self.att_weight_hdf.get_path(), allow_pickle=True)
            numpy_object = numpy_object[()]

            for key, object_dict in numpy_object.items():
                seq_tag = object_dict["tag"]
                targets = object_dict["classes"]
                att_weights = object_dict["rec_att_weights"][:, 0, :]  # [S, T]

                att_weights_dict[seq_tag] = att_weights
                targets_dict[seq_tag] = targets

        if self.seg_starts_hdf is not None:
            seg_starts_dict = hdf.load_hdf_data(self.seg_starts_hdf, segment_list=self.segment_whitelist)
            seg_lens_dict = hdf.load_hdf_data(self.seg_lens_hdf, segment_list=self.segment_whitelist)
        else:
            seg_starts_dict = None
            seg_lens_dict = None
        if self.center_positions_hdf is not None:
            center_positions_dict = hdf.load_hdf_data(self.center_positions_hdf, segment_list=self.segment_whitelist)
        else:
            center_positions_dict = None
        if self.ctc_alignment_hdf is not None:
            ctc_alignment_dict = hdf.load_hdf_data(self.ctc_alignment_hdf, segment_list=self.segment_whitelist)
        else:
            ctc_alignment_dict = None
        if self.ref_alignment_hdf is not None:
            ref_alignment_dict = hdf.load_hdf_data(self.ref_alignment_hdf, segment_list=self.segment_whitelist)
            ref_alignment_dict = {k: v for k, v in ref_alignment_dict.items() if k in att_weights_dict}
        else:
            ref_alignment_dict = {}

        return (
            att_weights_dict,
            targets_dict,
            seg_starts_dict,
            seg_lens_dict,
            center_positions_dict,
            ctc_alignment_dict,
            ref_alignment_dict,
        )

    @staticmethod
    def _get_fig_ax(att_weights_: List[np.ndarray], upsampling_factor: int = 1, scale: float = 1.0):
        """
        Initialize the figure and axis for the plot.
        """
        att_weights = att_weights_[0]

        num_labels = att_weights.shape[0]
        num_frames = att_weights.shape[1] // upsampling_factor
        # change figsize depending on number of frames and labels
        if num_frames == num_labels:
            fig_width = num_frames / 4
            fig_height = fig_width
        else:
            fig_width = 7 * num_frames / 50
            # y size is the number of labels times the font size times the scale factor
            fig_height = (
                num_labels * PlotAttentionWeightsJobV2.font_size_small * PlotAttentionWeightsJobV2.pt_to_inches * 1.05
                + 3.0
            ) * scale
            # if num_frames < 32:
            #   width_factor = 4
            #   height_factor = 2
            # else:
            #   width_factor = 8
            #   height_factor = 2
            # fig_width = num_frames / width_factor
            # fig_height = num_labels / height_factor

        if any([size < 2.0 for size in (fig_width, fig_height)]):
            factor = 2
        else:
            factor = 1

        figsize = (fig_width * factor, fig_height * factor)

        figsize = [figsize[0], figsize[1] * len(att_weights_)]
        # if len(att_weights_) > 1:
        # figsize[1] *= 0.6

        fig, axes = plt.subplots(
            len(att_weights_),
            figsize=figsize,
            # constrained_layout=True
        )

        if len(att_weights_) == 1:
            axes = [axes]

        return fig, axes

    @staticmethod
    def set_ticks(
        ax: plt.Axes,
        ref_alignment: Optional[np.ndarray],
        targets: np.ndarray,
        target_vocab: Dict[int, str],
        ref_vocab: Dict[int, str],
        ref_alignment_blank_idx: Optional[int],
        time_len: int,
        target_blank_idx: Optional[int] = None,
        draw_vertical_lines: bool = False,
        use_segment_center_as_label_position: bool = False,
        upsample_factor: int = 1,
        use_time_axis: bool = True,
        use_ref_axis: bool = True,
        scale: float = 1.0,
    ):
        """
        Set the ticks and labels for the x and y axis.
        x-axis: reference alignment
        y-axis: model output
        """

        if ref_alignment is not None:
            ref_label_positions = np.where(ref_alignment != ref_alignment_blank_idx)[-1] + 1
            vertical_lines = [tick - 1.0 for tick in ref_label_positions]

        if use_ref_axis:
            # positions of reference labels in the reference alignment
            # +1 bc plt starts at 1, not at 0
            ref_labels = ref_alignment[ref_alignment != ref_alignment_blank_idx]
            ref_labels = [ref_vocab[idx] for idx in ref_labels]
            # x axis
            if use_segment_center_as_label_position:
                ref_label_positions = np.concatenate([[0], ref_label_positions])
                ref_segment_sizes = ref_label_positions[1:] - ref_label_positions[:-1]
                xticks = ref_label_positions[:-1] + ref_segment_sizes / 2
            else:
                xticks = vertical_lines
            ax.set_xticks(xticks)
            ax.set_xticklabels(ref_labels, rotation=90, fontsize=PlotAttentionWeightsJobV2.font_size_small * scale)
            ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)

            # ax.set_xlabel("Reference Alignment", fontsize=14)
            # ax.xaxis.set_label_position('top')
        else:
            ax.set_xticks([])

        if use_time_axis:
            # secondary x-axis at the bottom with time stamps
            time_step_size = 50
            time_ticks = range(0, len(ref_alignment) if ref_alignment is not None else time_len, time_step_size)

            # Create the secondary y-axis at the bottom
            time_axis = ax.secondary_xaxis("bottom")

            # Set the ticks and labels for the secondary y-axis
            time_axis.set_xticks(time_ticks)
            xtick_labels = [(time_tick * 60 / upsample_factor) / 1000 for time_tick in time_ticks]
            time_axis.set_xticklabels(
                [f"{label:.1f}" for label in xtick_labels], fontsize=PlotAttentionWeightsJobV2.font_size_small * scale
            )
            # time_axis.set_xticks([])  # no time ticks

            time_axis.set_xlabel(
                "Time (s) ($\\rightarrow$)", fontsize=PlotAttentionWeightsJobV2.font_size_large * scale
            )
            # ----

        # output labels of the model
        # in case of alignment: filter for non-blank targets. otherwise, just leave the targets array as is
        labels = targets[targets != target_blank_idx] if target_blank_idx is not None else targets
        labels = [target_vocab[idx] for idx in labels]
        # y axis
        yticks = [tick for tick in range(len(labels))]
        ax.set_yticks(yticks)
        ax.set_yticklabels(labels, fontsize=PlotAttentionWeightsJobV2.font_size_small * scale)

        # horizontal lines to separate labels on y axis
        for ytick in yticks:
            ax.axhline(y=ytick - 0.5, xmin=0, xmax=1, color="k", linewidth=0.5)

        if (draw_vertical_lines or True) and ref_alignment is not None:  # always draw
            for xtick in vertical_lines:
                if len(ref_alignment) == len(targets):
                    # this is like a square grid
                    x = xtick + 0.5
                    linestyle = "-"
                else:
                    # here, the vertical lines are at the center of the reference labels
                    x = xtick
                    linestyle = "--"
                ax.axvline(x=x, ymin=0, ymax=1, color="k", linewidth=0.5, linestyle=linestyle, alpha=0.8)

        # axis labels
        # ax.set_ylabel("Output Labels ($\\rightarrow$)", fontsize=14)

    @staticmethod
    def _draw_segment_boundaries(
        ax: plt.Axes,
        seg_starts: np.ndarray,
        seg_lens: np.ndarray,
        att_weights: np.ndarray,
    ):
        """
        Draw red delimiters to indicate segment boundaries
        """
        num_labels = att_weights.shape[0]
        for i, (seg_start, seg_len) in enumerate(zip(seg_starts, seg_lens)):
            ymin = i / num_labels
            ymax = (i + 1) / num_labels
            ax.axvline(x=seg_start - 0.5, ymin=ymin, ymax=ymax, color="r")
            ax.axvline(x=min(seg_start + seg_len - 0.5, att_weights.shape[1] - 0.5), ymin=ymin, ymax=ymax, color="r")

    @staticmethod
    def _draw_center_positions(
        ax: plt.Axes,
        center_positions: np.ndarray,
        upsample_factor: int = 1,
    ):
        """
        Draw green delimiters to indicate center positions
        """
        num_labels = center_positions.shape[0]
        for i, center_position in enumerate(center_positions):
            ymin = i / num_labels
            ymax = (i + 1) / num_labels
            ax.axvline(x=center_position - 0.5, ymin=ymin, ymax=ymax, color="lime")
            ax.axvline(x=center_position + 0.5 + 1.0 * (upsample_factor - 1), ymin=ymin, ymax=ymax, color="lime")

    @staticmethod
    def plot_ctc_alignment(
        ax: plt.Axes, ctc_alignment: np.ndarray, num_labels: int, ctc_blank_idx: int, plot_trailing_blanks: bool = False
    ):
        label_idx = 0
        # store alignment like: 000011112222223333, where the number is the label index (~ height in the plot)
        ctc_alignment_plot_data = []
        for x, ctc_label in enumerate(ctc_alignment):
            ctc_alignment_plot_data.append(label_idx)
            if ctc_label != ctc_blank_idx:
                label_idx += 1
            # stop if we reached the last label, the rest of the ctc alignment are blanks
            if label_idx == num_labels and not plot_trailing_blanks:
                break
        ax.plot(ctc_alignment_plot_data, "o", color="black", alpha=0.4)

    @staticmethod
    def plot_center_of_gravity(ax: plt.Axes, att_weights: np.ndarray):
        cog = np.sum(att_weights * np.arange(att_weights.shape[1])[None, :], axis=1)[:, None]  # [S,1]
        ax.plot(cog, range(cog.shape[0]), "o", color="red", alpha=0.4)

    def run(self):
        # load data from hdfs
        (
            att_weights_dict,
            targets_dict,
            seg_starts_dict,
            seg_lens_dict,
            center_positions_dict,
            ctc_alignment_dict,
            ref_alignment_dict,
        ) = self.load_data()

        if self.other_att_weights is not None:
            other_att_weight_dicts = [
                hdf.load_hdf_data(att_weight_hdf, num_dims=2, segment_list=self.segment_whitelist)
                for att_weight_hdf in self.other_att_weights
            ]
        else:
            other_att_weight_dicts = []

        # load vocabulary as dictionary
        with open(self.json_vocab_path.get_path(), "r", encoding="utf-8") as f:
            json_data = f.read()
            target_vocab = ast.literal_eval(json_data)  # label -> idx
            # switch keys and values and replace EOS token by "$$\\texttt{EOS}$$" (as in my thesis)
            target_vocab = {v: k for k, v in target_vocab.items()}  # idx -> label
            # if we have a target blank idx, we replace the EOS symbol in the vocab with "<b>"
            if self.target_blank_idx is not None:
                target_vocab[self.target_blank_idx] = "<b>"

        # load reference alignment vocabulary as dictionary
        if self.ref_alignment_json_vocab_path != self.json_vocab_path:
            with open(self.ref_alignment_json_vocab_path.get_path(), "r") as f:
                json_data = f.read()
                ref_vocab = ast.literal_eval(json_data)
                ref_vocab = {v: k for k, v in ref_vocab.items()}
        else:
            ref_vocab = target_vocab

        # for each seq tag, plot the corresponding att weights
        for seq_tag in att_weights_dict.keys():
            seg_starts = seg_starts_dict[seq_tag] if self.seg_starts_hdf is not None else None  # [S]
            seg_lens = seg_lens_dict[seq_tag] if self.seg_lens_hdf is not None else None  # [S]
            center_positions = center_positions_dict[seq_tag] if self.center_positions_hdf is not None else None  # [S]
            ctc_alignment = ctc_alignment_dict[seq_tag] if self.ctc_alignment_hdf is not None else None  # [T]
            ref_alignment = ref_alignment_dict.get(seq_tag)  # [T]
            targets = targets_dict[seq_tag]  # [S]
            att_weights = att_weights_dict[seq_tag]  # [S,T]
            upsampling_factor = 1  # hard code for now

            if ref_alignment is None:
                self.ref_alignment_blank_idx = 0
                ref_alignment = np.zeros((att_weights.shape[1] * upsampling_factor,), dtype=np.int32)

            att_weights_list = [att_weights]
            for other_att_weight_dict in other_att_weight_dicts:
                other_att_weights = other_att_weight_dict[seq_tag]
                att_weights_list.append(other_att_weights)

            for i in range(len(att_weights_list)):
                # upsample attention weights by factor of six to get to 10ms frames

                att_weights_ = att_weights_list[i]

                if ref_alignment is not None:
                    # if ref alignment is given and has the same length as the att weights, we need to upsample it as well
                    # otherwise, we assume the ref alignment comes from a HMM and already has the correct length (10ms frames)
                    ref_align_att_weight_ratio = ref_alignment.shape[0] // att_weights_.shape[1]
                    if ref_align_att_weight_ratio == 1:
                        ref_alignment = ref_alignment[:, None]
                        blanks = np.full(
                            list(ref_alignment.shape) + [upsampling_factor - 1],
                            dtype=np.int32,
                            fill_value=self.ref_alignment_blank_idx,
                        )
                        ref_alignment = np.dstack((ref_alignment, blanks)).reshape(-1)

                # repeat each attention weight by upsample_factor times
                att_weights_ = np.repeat(att_weights_, upsampling_factor, axis=1)
                if att_weights_.shape[1] < ref_alignment.shape[0]:
                    assert ref_alignment.shape[0] - att_weights_.shape[1] < upsampling_factor
                    att_weights_ = np.concatenate(
                        [
                            att_weights_,
                            np.zeros((att_weights_.shape[0], ref_alignment.shape[0] - att_weights_.shape[1]))
                            + np.min(att_weights_),
                        ],
                        axis=1,
                    )
                # cut off the last frames if necessary
                att_weights_ = att_weights_[:, : ref_alignment.shape[0]]

                # apply upsampling to segment starts, segment lengths and center positions
                if seg_lens is not None:
                    seg_lens = seg_lens * upsampling_factor
                if seg_starts is not None:
                    seg_starts = seg_starts * upsampling_factor
                if center_positions is not None:
                    center_positions = center_positions * upsampling_factor

                # if ref_alignment is None:
                #   upsampling_factor = 1
                # else:
                #   upsampling_factor = ref_alignment.shape[0] // att_weights_.shape[1]
                # if upsampling_factor != 1:
                #   upsampling_factor = 6  # hard code for now
                #   # repeat each frame by upsample_factor times
                #   att_weights_ = np.repeat(att_weights_, upsampling_factor, axis=1)
                #   if att_weights_.shape[1] < ref_alignment.shape[0]:
                #     assert ref_alignment.shape[0] - att_weights_.shape[1] < upsampling_factor
                #     att_weights_ = np.concatenate([att_weights_, np.zeros((att_weights_.shape[0], ref_alignment.shape[0] - att_weights_.shape[1])) + np.min(att_weights_)], axis=1)
                #   # cut off the last frames if necessary
                #   att_weights_ = att_weights_[:, :ref_alignment.shape[0]]

                att_weights_list[i] = att_weights_

            fig, axes = self._get_fig_ax(
                att_weights_list,
                upsampling_factor=(upsampling_factor // 2) if upsampling_factor > 1 else 1,
                scale=self.scale,
            )

            # vmin = None if self.vmin is None else self.vmin[seq_tag]
            vmin = self.vmin
            if isinstance(vmin, dict):
                vmin = vmin[seq_tag]

            # vmax = None if self.vmax is None else self.vmax[seq_tag]
            vmax = self.vmax
            if isinstance(vmax, dict):
                vmax = vmax[seq_tag]

            for i, ax in enumerate(axes):
                att_weights = att_weights_list[i]
                mat = ax.matshow(
                    att_weights,
                    cmap=plt.cm.get_cmap("Greys") if self.plot_w_color_gradient else plt.cm.get_cmap("Blues"),
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                )

                if self.titles is not None:
                    assert len(self.titles) == len(axes)
                    ax.set_title(
                        self.titles[i], fontsize=PlotAttentionWeightsJobV2.font_size_large * self.scale, pad=20
                    )

                if self.plot_w_color_gradient:
                    gradient = np.linspace(0, 1, att_weights.shape[1])
                    gradient = np.repeat(gradient[None, :], att_weights.shape[0], axis=0)
                    ax.imshow(gradient, aspect="auto", cmap="hsv", alpha=0.5, interpolation=None)

                use_segment_center_as_label_position = self.ref_alignment_hdf is not None and (
                    "GmmAlignmentToWordBoundaries" in self.ref_alignment_hdf.get_path()
                )
                if self.ref_alignment_is_positional_alignment is not None:
                    use_segment_center_as_label_position = not self.ref_alignment_is_positional_alignment

                # set y ticks and labels
                self.set_ticks(
                    ax=ax,
                    ref_alignment=ref_alignment,
                    targets=targets,
                    target_vocab=target_vocab,
                    ref_vocab=ref_vocab,
                    ref_alignment_blank_idx=self.ref_alignment_blank_idx,
                    target_blank_idx=self.target_blank_idx,
                    # a bit ugly but this is currently the only situation where we have a segmental ref alignment
                    use_segment_center_as_label_position=use_segment_center_as_label_position,
                    upsample_factor=upsampling_factor,
                    use_time_axis=i == len(axes) - 1,
                    use_ref_axis=i == 0 and ref_alignment is not None,
                    time_len=att_weights.shape[1],
                    scale=self.scale,
                )
                if seg_starts is not None:
                    self._draw_segment_boundaries(ax, seg_starts, seg_lens, att_weights)
                if center_positions is not None:
                    self._draw_center_positions(ax, center_positions, upsample_factor=upsampling_factor)

                ax.invert_yaxis()

                # use individual colorbars if vmin and vmax are not set
                if self.vmin is None or self.vmax is None:
                    cax = fig.add_axes(
                        [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height]
                    )
                    cbar = plt.colorbar(mat, cax=cax)
                    cbar.ax.tick_params(labelsize=16)

            # use single colorbar if vmin and vmax are set
            if self.vmin is not None and self.vmax is not None:
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes([0.91, 0.5 - (0.4 / len(axes)), 0.02, 0.8 / len(axes)])
                cbar = fig.colorbar(mat, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=self.font_size_medium * self.scale)

            fig.text(
                0.02,
                0.5,
                "Output Labels ($\\rightarrow$)",
                va="center",
                rotation="vertical",
                fontsize=PlotAttentionWeightsJobV2.font_size_large * self.scale,
            )
            # left, width = .25, .5
            # bottom, height = .25, .5
            # top = bottom + height
            # fig.text(
            #   left, 0.5 * (bottom + top),
            #   'Output Labels ($\\rightarrow$)',
            #   horizontalalignment='right',
            #   verticalalignment='center',
            #   rotation='vertical',
            #   transform=plt.gca().transAxes,
            # )

            dirname = self.out_plot_dir.get_path()
            if seq_tag.startswith("dev-other"):
                concat_num = len(seq_tag.split(";"))
                if concat_num > 1:
                    filename = os.path.join(
                        dirname, f"plot.{seq_tag.split(';')[0].replace('/', '_')}.+{concat_num - 1}"
                    )
                else:
                    filename = os.path.join(dirname, "plot.%s" % seq_tag.replace("/", "_"))
            else:
                filename = os.path.join(dirname, "plot.%s" % seq_tag.replace("/", "_"))

            plt.savefig(filename + ".png", bbox_inches="tight")
            plt.savefig(filename + ".pdf", bbox_inches="tight")

            # plot ctc alignment if available
            if ctc_alignment is not None:
                for i, ax in enumerate(axes):
                    self.plot_ctc_alignment(
                        ax,
                        ctc_alignment,
                        num_labels=att_weights.shape[0],
                        ctc_blank_idx=self.ref_alignment_blank_idx,  # works for us but better to add separate attribute for ctc
                    )
                filename = os.path.join(self.out_plot_w_ctc_dir.get_path(), "plot.%s" % seq_tag.replace("/", "_"))
                plt.savefig(filename + ".png")
                plt.savefig(filename + ".pdf")

            # plot center of gravity
            if self.plot_w_cog:
                for i, ax in enumerate(axes):
                    ax.lines[-1].remove()  # remove ctc alignment plot
                    self.plot_center_of_gravity(ax, att_weights)
                filename = os.path.join(self.out_plot_w_cog_dir.get_path(), "plot.%s" % seq_tag.replace("/", "_"))
                plt.savefig(filename + ".png")
                plt.savefig(filename + ".pdf")

            plt.close()

    @classmethod
    def hash(cls, kwargs: Dict[str, Any]):
        d = copy.deepcopy(kwargs)

        if d["center_positions_hdf"] is None:
            d.pop("center_positions_hdf")
        if d["ctc_alignment_hdf"] is None:
            d.pop("ctc_alignment_hdf")
        if d["ref_alignment_json_vocab_path"] is None:
            d.pop("ref_alignment_json_vocab_path")
        if d["plot_w_cog"] is True:
            d.pop("plot_w_cog")
        if d["titles"] is None:
            d.pop("titles")
        if d["vmin"] == 0.0:
            d.pop("vmin")
        if d["vmax"] == 1.0:
            d.pop("vmax")
        if d["scale"] == 1.0:
            d.pop("scale")
        if d["ref_alignment_is_positional_alignment"] is None:
            d.pop("ref_alignment_is_positional_alignment")

        d["version"] = 2

        return super().hash(d)
