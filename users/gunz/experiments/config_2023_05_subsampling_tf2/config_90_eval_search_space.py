__all__ = ["run"]

import dataclasses
import os

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk, Path

from ...setups.common.rasr.search_space import AllophoneDetails, VisualizeBestTraceJob
from ...setups.fh.factored import LabelInfo, RasrStateTying


def run():
    # ******************** Settings ********************
    gs.ALIAS_AND_OUTPUT_SUBDIR = os.path.splitext(os.path.basename(__file__))[0][7:]

    # ********************* Plots **********************

    li = dataclasses.replace(LabelInfo.default_ls(), state_tying=RasrStateTying.monophone, n_states_per_phone=3)
    li_ss = dataclasses.replace(li, n_states_per_phone=1)

    segments = [
        "dev-other/116-288045-0015",
        "dev-other/116-288045-0025",
        "dev-other/116-288046-0005",
        "dev-other/6123-59150-0012",
        "dev-other/6123-59150-0006",
    ]
    non_ss = VisualizeBestTraceJob(
        rasr_logs=Path("/u/mgunz/gunz/ma/state-space-comparison/recog-normal/rasr.log"),
        state_tying=Path("/u/mgunz/gunz/ma/state-space-comparison/monophone-dense-state-tying-n3"),
        num_tied_phonemes=li.get_n_of_dense_classes(),
        segments_to_process=segments,
        allophone_detail_level=AllophoneDetails.MONOPHONE,
        x_steps_per_log_time_step=1,
    )
    alias_output_visualize(non_ss, "non-ss")

    ss = VisualizeBestTraceJob(
        rasr_logs=Path("/u/mgunz/gunz/ma/state-space-comparison/recog-ss/rasr.log"),
        state_tying=Path("/u/mgunz/gunz/ma/state-space-comparison/monophone-dense-state-tying-n1"),
        num_tied_phonemes=li_ss.get_n_of_dense_classes(),
        segments_to_process=segments,
        allophone_detail_level=AllophoneDetails.MONOPHONE,
        x_steps_per_log_time_step=4,
    )
    alias_output_visualize(ss, "ss-40ms")


def alias_output_visualize(j: VisualizeBestTraceJob, name: str):
    j.add_alias(f"search-space/{name}")

    for i, out in enumerate(j.out_print_files.values()):
        tk.register_output(f"non-ss/out.{i}.txt", out)
    for i, out in enumerate(j.out_plot_files.values()):
        tk.register_output(f"non-ss/out.{i}.png", out)
