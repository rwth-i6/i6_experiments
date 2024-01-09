import os

from sisyphus import gs, tk, Path

from ...setups.common.analysis import (
    ComputeAlignmentSamplingStatisticsJob,
    ComputeSilencePercentageJob,
    ComputeTimestampErrorJob,
    ComputeWordLevelTimestampErrorJob,
    PlotPhonemeDurationsJob,
    PlotViterbiAlignmentsJob,
)
from ...setups.common.analysis.tse_dmann import DMannComputeTseJob
from ...setups.common.analysis.tse_tina import ComputeTinaTseJob
from ..config_2023_05_baselines_thesis_tf2.config import SCRATCH_ALIGNMENT
from .config import (
    ALIGN_GMM_MONO_10MS,
    ALIGN_GMM_TRI_10MS,
    ALIGN_GMM_TRI_ALLOPHONES,
    ZHOU_ALLOPHONES,
    ZHOU_SUBSAMPLED_ALIGNMENT,
)


def map_seg_tag(ref_tag: str) -> str:
    # from train-other-960/103-1240-0000/103-1240-0000
    # to librispeech/8425-292520/0013

    crp, tag, _ = ref_tag.split("/")
    a, b, c = tag.split("-")
    return f"librispeech/{a}-{b}/{c}"


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
        font_size=25,
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/40ms-zhou-blstm/alignment-plots", plots.out_plot_folder)
    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True),
        allophones_path=Path(ZHOU_ALLOPHONES),
        segments=["librispeech/2920-156224/0013"],
        font_size=25,
        show_labels=False,
        show_title=False,
        monophone=True,
    )
    tk.register_output(f"alignments/40ms-zhou-blstm/alignment-plots-plain", plots.out_plot_folder)

    tse_tina_job = ComputeTinaTseJob(
        allophones=tk.Path(ZHOU_ALLOPHONES),
        alignment_bundle=tk.Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True),
        ref_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        ref_alignment_bundle=tk.Path(ALIGN_GMM_MONO_10MS, cached=True),
        ref_t_step=10 / 1000,
        ss_factor=4,
        map_seg_tags=map_seg_tag,
    )
    tk.register_output(f"alignments/40ms-zhou-blstm/statistics/tse-tina", tse_tina_job.out_tse)

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

    for n in [3, 4]:
        sampled = ComputeAlignmentSamplingStatisticsJob(
            alignment_bundle=Path(SCRATCH_ALIGNMENT, cached=True),
            allophone_file=Path(
                "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
            ),
            sample_rate=n,
        )
        tk.register_output(f"alignments/10ms-scratch-blstm/statistics/sample-{n}x/total", sampled.out_total_phones)
        tk.register_output(f"alignments/10ms-scratch-blstm/statistics/sample-{n}x/skipped", sampled.out_total_skipped)
        tk.register_output(f"alignments/10ms-scratch-blstm/statistics/sample-{n}x/ratio", sampled.out_ratio_skipped)
        tk.register_output(
            f"alignments/10ms-scratch-blstm/statistics/sample-{n}x/segments", sampled.out_segments_with_sampling
        )

    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(SCRATCH_ALIGNMENT, cached=True),
        allophones_path=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        segments=[
            "train-other-960/2920-156224-0013/2920-156224-0013",
            "train-other-960/2498-134786-0003/2498-134786-0003",
            "train-other-960/6178-86034-0008/6178-86034-0008",
            "train-other-960/5983-39669-0034/5983-39669-0034",
        ],
        font_size=25,
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/alignment-plots", plots.out_plot_folder)
    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(SCRATCH_ALIGNMENT, cached=True),
        allophones_path=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        segments=[
            "train-other-960/2920-156224-0013/2920-156224-0013",
            "train-other-960/2498-134786-0003/2498-134786-0003",
            "train-other-960/6178-86034-0008/6178-86034-0008",
            "train-other-960/5983-39669-0034/5983-39669-0034",
        ],
        font_size=25,
        show_labels=False,
        show_title=False,
        monophone=True,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/alignment-plots-plain", plots.out_plot_folder)

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

    tse_tina_job = ComputeTinaTseJob(
        allophones=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        alignment_bundle=Path(SCRATCH_ALIGNMENT, cached=True),
        ref_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        ref_alignment_bundle=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        ref_t_step=10 / 1000,
        ss_factor=1,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/statistics/tse-tina", tse_tina_job.out_tse)

    tse_tina_job = ComputeTinaTseJob(
        allophones=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        alignment_bundle=Path(SCRATCH_ALIGNMENT, cached=True),
        ref_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        ref_alignment_bundle=tk.Path(ALIGN_GMM_MONO_10MS, cached=True),
        ref_t_step=10 / 1000,
        ss_factor=1,
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/statistics/tse-tina-mono", tse_tina_job.out_tse)

    dmann_tse = DMannComputeTseJob(
        allophones=Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
        alignment=Path(SCRATCH_ALIGNMENT, cached=True),
        reference_alignment=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/tse_dmann", dmann_tse.out_tse)

    sil_job = ComputeSilencePercentageJob(
        Path(SCRATCH_ALIGNMENT, cached=True),
        Path(
            "/work/asr3/raissi/shared_workspaces/gunz/2023-05--subsampling-tf2/i6_core/lexicon/allophones/StoreAllophonesJob.Qa3bLX1BHz42/output/allophones"
        ),
    )
    tk.register_output(f"alignments/10ms-scratch-blstm/sil", sil_job.out_percent_sil)

    # GMM

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
        segments=[
            "train-other-960/2920-156224-0013/2920-156224-0013",
            "train-other-960/2498-134786-0003/2498-134786-0003",
            "train-other-960/6178-86034-0008/6178-86034-0008",
            "train-other-960/5983-39669-0034/5983-39669-0034",
        ],
        font_size=25,
        show_labels=False,
        monophone=True,
    )
    tk.register_output(f"alignments/10ms-gmm-tri/alignment-plots", plots.out_plot_folder)
    plots = PlotViterbiAlignmentsJob(
        alignment_bundle_path=Path(ALIGN_GMM_TRI_10MS, cached=True),
        allophones_path=Path(ALIGN_GMM_TRI_ALLOPHONES),
        segments=[
            "train-other-960/2920-156224-0013/2920-156224-0013",
            "train-other-960/2498-134786-0003/2498-134786-0003",
            "train-other-960/6178-86034-0008/6178-86034-0008",
            "train-other-960/5983-39669-0034/5983-39669-0034",
        ],
        font_size=25,
        show_labels=False,
        show_title=False,
        monophone=True,
    )
    tk.register_output(f"alignments/10ms-gmm-tri/alignment-plots-plain", plots.out_plot_folder)
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

    sil_job = ComputeSilencePercentageJob(tk.Path(ALIGN_GMM_TRI_10MS, cached=True), tk.Path(ALIGN_GMM_TRI_ALLOPHONES))
    tk.register_output(f"alignments/10ms-gmm-tri/sil", sil_job.out_percent_sil)

    tse_tina_job = ComputeTinaTseJob(
        allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        alignment_bundle=tk.Path(ALIGN_GMM_TRI_10MS, cached=True),
        ref_allophones=tk.Path(ALIGN_GMM_TRI_ALLOPHONES),
        ref_alignment_bundle=tk.Path(ALIGN_GMM_MONO_10MS, cached=True),
        ref_t_step=10 / 1000,
        ss_factor=1,
    )
    tk.register_output(f"alignments/10ms-gmm-tri/statistics/tse-tina-mono", tse_tina_job.out_tse)
