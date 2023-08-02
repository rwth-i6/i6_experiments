import os

from sisyphus import gs, tk, Path

from ...setups.common.analysis import PlotPhonemeDurationsJob
from .config import ZHOU_ALLOPHONES, ZHOU_SUBSAMPLED_ALIGNMENT


def run():
    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]

    plots = PlotPhonemeDurationsJob(
        alignment_bundle_path=Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True),
        allophones_path=Path(ZHOU_ALLOPHONES),
        time_step_s=40 / 1000,
    )
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/plots", plots.out_plot_folder)
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/means", plots.out_means)
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/variances", plots.out_vars)
