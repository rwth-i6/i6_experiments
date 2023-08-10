import os

from sisyphus import gs, tk, Path

from ...setups.common.analysis import PlotPhonemeDurationsJob
from ..config_2023_05_baselines_thesis_tf2.config import SCRATCH_ALIGNMENT
from .config import ZHOU_ALLOPHONES, ZHOU_SUBSAMPLED_ALIGNMENT


def run():
    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]

    zhou_data = PlotPhonemeDurationsJob(
        alignment_bundle_path=Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True),
        allophones_path=Path(ZHOU_ALLOPHONES),
        time_step_s=40 / 1000,
    )
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/plots", zhou_data.out_plot_folder)
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/means", zhou_data.out_means)
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/variances", zhou_data.out_vars)

    scratch_data = PlotPhonemeDurationsJob(
        alignment_bundle_path=Path(SCRATCH_ALIGNMENT, cached=True),
        allophones_path=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        time_step_s=10 / 1000,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/statistics/plots", scratch_data.out_plot_folder)
    tk.register_output(f"alignments/10ms-scratch-blstm/statistics/means", scratch_data.out_means)
    tk.register_output(f"alignments/10ms-scratch-blstm/statistics/variances", scratch_data.out_vars)
