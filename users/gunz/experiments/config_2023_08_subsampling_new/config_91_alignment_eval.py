import os

from sisyphus import gs, tk, Path

from ...setups.common.analysis import (
    ComputeTimestampErrorJob,
    ComputeWordLevelTimestampErrorJob,
    PlotPhonemeDurationsJob,
    PlotViterbiAlignmentsJob,
)
from ...setups.common.analysis.tse_dmann import DMannComputeTseJob
from ...setups.common.util import ComputeAverageJob
from ..config_2023_05_baselines_thesis_tf2.config import SCRATCH_ALIGNMENT
from .config import ALIGN_GMM_TRI_10MS, ALIGN_GMM_TRI_ALLOPHONES, ZHOU_ALLOPHONES, ZHOU_SUBSAMPLED_ALIGNMENT


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

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True),
        allophones_path=Path(ZHOU_ALLOPHONES),
        segments=["librispeech/2920-156224/0013"],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/40ms-zhou-blstm/alignment-plots", plots.out_plot_folder)

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

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(SCRATCH_ALIGNMENT, cached=True),
        allophones_path=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        segments=["train-other-960/2920-156224-0013/2920-156224-0013"],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/alignment-plots", plots.out_plot_folder)

    tse_job = ComputeTimestampErrorJob(
        allophones=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        alignment=Path(SCRATCH_ALIGNMENT, cached=True),
        t_step=10 / 1000,
        reference_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        reference_t_step=10 / 1000,
        fuzzy_match_mismatching_phoneme_sequences=False,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/tse", tse_job.out_tse)

    tse_w_job = ComputeWordLevelTimestampErrorJob(
        allophones=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        alignment=Path(SCRATCH_ALIGNMENT, cached=True),
        t_step=10 / 1000,
        reference_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        reference_t_step=10 / 1000,
        fuzzy_match_mismatching_phoneme_sequences=False,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/statistics/tse-w", tse_w_job.out_tse)

    dmann_tse = DMannComputeTseJob(
        allophones=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        alignment=Path(SCRATCH_ALIGNMENT, cached=True),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/tse_dmann", dmann_tse.out_tse)

    scratch_data = PlotPhonemeDurationsJob(
        alignment_bundle_path=Path(ALIGN_GMM_TRI_10MS, cached=True),
        allophones_path=Path(ALIGN_GMM_TRI_ALLOPHONES),
        time_step_s=10 / 1000,
    )
    tk.register_output(f"alignments/10ms-gmm-tri/statistics/plots", scratch_data.out_plot_folder)
    tk.register_output(f"alignments/10ms-gmm-tri/statistics/means", scratch_data.out_means)
    tk.register_output(f"alignments/10ms-gmm-tri/statistics/variances", scratch_data.out_vars)

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(ALIGN_GMM_TRI_10MS, cached=True),
        allophones_path=Path(ALIGN_GMM_TRI_ALLOPHONES),
        segments=["train-other-960/2920-156224-0013/2920-156224-0013"],
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/10ms-gmm-tri/alignment-plots", plots.out_plot_folder)
    tse_job = ComputeTimestampErrorJob(
        allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        t_step=10 / 1000,
        reference_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        reference_t_step=10 / 1000,
        fuzzy_match_mismatching_phoneme_sequences=False,
    )
    tk.register_output(f"alignments/10ms-gmm-tri/tse", tse_job.out_tse)
