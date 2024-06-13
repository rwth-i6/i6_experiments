import copy
from typing import List
from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey

from .config_01a_ctc_blstm import py as py_01a
from .config_01b_ctc_conformer import py as py_01b
from .config_01c_ctc_blstm_wei_data import py as py_01c
from .config_02b_transducer_wei_data import py as py_02b
from .config_02c_transducer_wei_data_tinaconf import py as py_02c
from .config_02d_transducer_wei_data_am_scales import py as py_02d
from .config_03b_transducer_fullsum_wei_data import py as py_03b


def main() -> SummaryReport:
    summary_report = SummaryReport()

    for subreport in [
        py_01a()[0],
        py_01b()[0],
        py_01c()[0],
        py_02b()[0],
        py_02c()[0],
        py_02d()[0],
        py_03b(),
    ]:
        subreport = copy.deepcopy(subreport)
        subreport.collapse(
            [SummaryKey.CORPUS.value], best_selector_key=SummaryKey.ERR.value
        )  # Keep one row for each recognition corpus
        summary_report.merge_report(subreport, update_structure=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
