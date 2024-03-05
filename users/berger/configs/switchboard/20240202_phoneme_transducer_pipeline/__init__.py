import copy
from typing import List
from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey

# from .config_01a_ctc_blstm import py as py_01a
# from .config_01b_ctc_conformer import py as py_01b
from .config_01c_ctc_blstm_wei_data import py as py_01c
from .config_02b_transducer_wei_data import py as py_02b


def main() -> SummaryReport:
    # gmm_config.run_librispeech_960_common_baseline()

    summary_report = SummaryReport()

    subreports: List[SummaryReport] = []

    subreports.append(copy.deepcopy(py_01c()[0]))
    subreports.append(copy.deepcopy(py_02b()[0]))

    for report in subreports:
        report.collapse(non_collapsed_keys=[SummaryKey.TRAIN_NAME.value])
        summary_report.merge_report(report, update_structure=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
