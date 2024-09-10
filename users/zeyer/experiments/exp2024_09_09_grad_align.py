"""
Alignments
"""

from __future__ import annotations
import os
import sys
from sisyphus import tk, Job, Task, Path


def py():
    # * CalculateSilenceStatistics:
    # der job holt sich die word end positions von dem GMM alignment und rechnet die initial und final silence aus
    # * ForcedAlignOnScoreMatrixJob
    tk.register_output(
        "grad-align",
        ForcedAlignOnScoreMatrixJob(
            # example (already in logspace):
            # score_matrix_hdf=Path(
            #     "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2/returnn_decoding/epoch-130-checkpoint/no-lm/beam-size-12/dev-other/analysis/analyze_gradients_ground-truth/3660-6517-0005_6467-62797-0001_6467-62797-0002_7697-105815-0015_7697-105815-0051/work/x_linear/log-prob-grads_wrt_x_linear_log-space/att_weights.hdf"
            # ),
            # non flipped grads
            score_matrix_hdf=Path(
                "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/forward/ReturnnForwardJobV2.KKMedG4R3uf4/output/gradients.hdf"
            ),
            plot=True,
            num_seqs=10,
        ).out_align,
    )


class ForcedAlignOnScoreMatrixJob(Job):
    def __init__(
        self,
        *,
        score_matrix_hdf: Path,
        apply_log: bool = True,
        plot: bool = False,
        num_seqs: int = -1,
    ):
        self.score_matrix_hdf = score_matrix_hdf
        self.apply_log = apply_log
        self.plot = plot
        self.num_seqs = num_seqs

        self.out_align = self.output_path("out_align")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        from typing import List, Tuple
        import numpy as np
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        from i6_experiments.users.schmitt.hdf import load_hdf_data

        score_matrix_data_dict = load_hdf_data(self.score_matrix_hdf, num_dims=2)

        for i, seq_tag in enumerate(score_matrix_data_dict):
            if 0 < self.num_seqs <= i:
                break

            print("seq tag:", seq_tag)
            apply_log_softmax = False
            plot_dir = f"alignments-{'w' if apply_log_softmax else 'wo'}-softmax"
            os.makedirs(plot_dir, exist_ok=True)

            score_matrix = score_matrix_data_dict[seq_tag]  # [S, T]
            # Last row is EOS, remove it.
            score_matrix = score_matrix[:-1]

            if self.apply_log:
                # Assuming L2 norm scores (i.e. >0).
                score_matrix = np.log(score_matrix)
            # Otherwise assume already in log space.
            # Make sure they are all negative or zero max.
            m = np.max(score_matrix)
            print("score matrix max:", m)
            score_matrix = score_matrix - max(m, 0.0)
            # score_matrix = -np.abs(score_matrix)
            # score_matrix = np.exp(score_matrix)
            if apply_log_softmax:
                max_score = np.max(score_matrix, axis=1, keepdims=True)
                score_matrix = score_matrix - max_score
                score_matrix = score_matrix - np.log(np.sum(np.exp(score_matrix), axis=1, keepdims=True))
            T = score_matrix.shape[1]  # noqa
            S = score_matrix.shape[0]  # noqa

            # scores/backpointers over the states and time steps.
            # states = blank/sil + labels. whether we give scores to blank (and what score) or not is to be configured.
            # [T, S*2+1]
            backpointers = np.full(
                (T, S * 2 + 1), 3, dtype=np.int32
            )  # 0: diagonal-skip, 1: diagonal, 2: left, 3: undefined
            align_scores = np.full((T, S * 2 + 1), -np.infty, dtype=np.float32)

            score_matrix_ = np.zeros((T, S * 2 + 1), dtype=np.float32)  # [T, S*2+1]
            score_matrix_[:, 1::2] = score_matrix.T
            score_matrix_[:, 0::2] = 0.0  # blank score

            # The first two states are valid start states.
            align_scores[0, :2] = score_matrix_[0, :2]
            backpointers[0, :] = 0  # doesn't really matter

            # calculate align_scores and backpointers
            for t in range(1, T):
                scores_diagonal_skip = np.full([2 * S + 1], -np.infty)
                scores_diagonal_skip[2:] = align_scores[t - 1, :-2] + score_matrix_[t, 2:]  # [2*S-1]
                scores_diagonal_skip[::2] = -np.infty  # diagonal skip is not allowed in blank
                scores_diagonal = np.full([2 * S + 1], -np.infty)
                scores_diagonal[1:] = align_scores[t - 1, :-1] + score_matrix_[t, 1:]  # [2*S]
                scores_horizontal = align_scores[t - 1, :] + score_matrix_[t, :]  # [2*S+1]

                score_cases = np.stack([scores_diagonal_skip, scores_diagonal, scores_horizontal], axis=0)  # [3, 2*S+1]
                backpointers[t] = np.argmax(score_cases, axis=0)  # [2*S+1]->[0,1,2]
                align_scores[t : t + 1] = np.take_along_axis(score_cases, backpointers[t : t + 1], axis=0)  # [1,2*S+1]

            # All but the last two states are not valid final states.
            align_scores[-1, :-2] = -np.infty

            # backtrace
            best_final = np.argmax(align_scores[-1])  # scalar, S*2 or S*2-1
            s = best_final
            t = T - 1
            alignment: List[Tuple[int, int]] = []
            while True:
                assert s >= 0 and t >= 0
                alignment.append((s, t))
                if t == 0:
                    assert s <= 1  # we should have reached the start states
                    break

                b = backpointers[t, s]
                if b == 0:
                    s -= 2
                    t -= 1
                elif b == 1:
                    s -= 1
                    t -= 1
                elif b == 2:
                    t -= 1
                else:
                    raise ValueError(f"invalid backpointer {b} at s={s}, t={t}")

            # TODO store in out hdf

            if self.plot:
                from matplotlib import pyplot as plt
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                alignment_map = np.zeros([T, 2 * S + 1], dtype=np.int32)  # [T, S*2+1]
                for s, t in alignment:
                    alignment_map[t, s] = 2 if s % 2 == 1 else 1

                fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
                for i, (alias, mat) in enumerate(
                    [
                        ("log(gradients) (local scores d)", score_matrix.T),
                        ("Partial scores D", -1 * align_scores),
                        ("backpointers", -1 * backpointers),
                        ("alignment", alignment_map),
                    ]
                ):
                    # mat is [T,S*2+1] or [T,S]
                    mat_ = ax[i].matshow(mat.T, cmap="Blues", aspect="auto")
                    ax[i].set_title(f"{alias} for seq {seq_tag}")
                    ax[i].set_xlabel("time")
                    ax[i].set_ylabel("labels")

                    divider = make_axes_locatable(ax[i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    if alias == "backpointers":
                        cbar = fig.colorbar(mat_, cax=cax, orientation="vertical", ticks=[0, -1, -2, -3])
                        cbar.ax.set_yticklabels(["diagonal-skip", "diagonal", "left", "unreachable"])
                    elif alias == "alignment":
                        cbar = fig.colorbar(mat_, cax=cax, orientation="vertical", ticks=[0, 1, 2])
                        cbar.ax.set_yticklabels(["", "blank", "label"])
                    else:
                        fig.colorbar(mat_, cax=cax, orientation="vertical")

                plt.tight_layout()
                plt.savefig(f"{plot_dir}/alignment_{seq_tag.replace('/', '_')}.png")
