import os

from sisyphus import gs, tk, Path

from ...setups.common.analysis import PlotViterbiAlignmentsJob
from .config import GMM_TRI_ALIGNMENT, SCRATCH_ALIGNMENT


def run():
    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]

    for a, a_name in [
        (Path(GMM_TRI_ALIGNMENT, cached=True), "gmm-tri"),
        (Path(SCRATCH_ALIGNMENT, cached=True), "scratch"),
    ]:
        plots = PlotViterbiAlignmentsJob(
            alignment_bundle_path=a,
            allophones_path=Path(
                "/u/mgunz/gunz/kept-experiments/2023-05--subsampling-tf2/alignments/30ms/conf-1-lr-v6-ss-3-fs-3-bw-0.3-pC-0.6-tdp-0.1-v2/allophones"
            ),
            segments=["train-other-960/2920-156224-0013/2920-156224-0013"],
            show_labels=False,
            monophone=True,
        )
        tk.register_output(f"alignments/{a_name}/alignment-plots", plots.out_plot_folder)
