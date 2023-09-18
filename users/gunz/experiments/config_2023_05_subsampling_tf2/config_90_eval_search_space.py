__all__ = ["run"]

import dataclasses
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

from ...setups.common.rasr.search_space import AllophoneDetails, VisualizeBestTraceJob, TraceSource
from ...setups.fh.factored import LabelInfo, RasrStateTying


def run():
    # ******************** Settings ********************
    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]

    # ********************* Plots **********************

    print("search-space analysis")

    li = dataclasses.replace(LabelInfo.default_ls(), state_tying=RasrStateTying.monophone, n_states_per_phone=3)
    li_ss = dataclasses.replace(li, n_states_per_phone=1)

    segments = [
        "dev-other/116-288045-0015/116-288045-0015",
        "dev-other/116-288045-0025/116-288045-0025",
        "dev-other/116-288046-0005/116-288046-0005",
        "dev-other/6123-59150-0012/6123-59150-0012",
        "dev-other/6123-59150-0006/6123-59150-0006",
    ]
    sources = [
        TraceSource(
            name="10ms",
            rasr_log=Path("/u/mgunz/gunz/ma/state-space-comparison/recog-normal/rasr.log"),
            state_tying=Path("/u/mgunz/gunz/ma/state-space-comparison/monophone-dense-state-tying-n3"),
            num_tied_phonemes=li.get_n_of_dense_classes(),
            x_steps_per_time_step=1,
        ),
        TraceSource(
            name="30ms",
            rasr_log=Path("/u/mgunz/gunz/ma/state-space-comparison/recog-ss/rasr.log"),
            state_tying=Path("/u/mgunz/gunz/ma/state-space-comparison/monophone-dense-state-tying-n1"),
            num_tied_phonemes=li_ss.get_n_of_dense_classes(),
            x_steps_per_time_step=4,
        ),
    ]
    j = VisualizeBestTraceJob(
        segments=segments,
        sources=sources,
        allophone_detail_level=AllophoneDetails.MONOPHONE,
    )
    j.add_alias("search-space/analysis")
    for seg, img in j.out_plot_files.items():
        tk.register_output(f"search-space/{seg}.png", img)
    for (seg, s), txt in j.out_print_files.items():
        tk.register_output(f"search-space/{seg}.{s}.txt", txt)
