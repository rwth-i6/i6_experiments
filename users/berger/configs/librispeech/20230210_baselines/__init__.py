from typing import List
from sisyphus import tk
import copy
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey

from .config_01a_ctc_blstm_raw_samples import py as py_01a
from .config_01b_ctc_blstm_rasr_features import py as py_01b
from .config_01c_ctc_conformer_raw_samples import py as py_01c
from .config_01d_ctc_conformer_rasr_features import py as py_01d
from .config_01e_ctc_blstm_rasr_features_wei_lex import py as py_01e
from .config_01f_ctc_conformer_rasr_features_wei_lex import py as py_01f
from .config_02a_transducer_raw_samples import py as py_02a
from .config_02b_transducer_rasr_features import py as py_02b
from .config_02c_transducer_rasr_features_wei_lex import py as py_02c
from .config_02e_transducer_rasr_features_tinaconf import py as py_02e
from .config_02e_transducer_rasr_features_tinaconf_rtf import py as py_02e_rtf
from .config_02f_transducer_rasr_features_am_scales import py as py_02f
from .config_03a_transducer_fullsum_raw_samples import py as py_03a
from .config_03b_transducer_fullsum_rasr_features import py as py_03b
from .config_03c_transducer_fullsum_rasr_features_wei_lex import py as py_03c
from .config_04b_transducer_fullsum_from_scratch_rasr_features import py as py_04b


def main() -> SummaryReport:
    summary_report = SummaryReport()

    sub_reports: List[SummaryReport] = []

    sub_reports.append(copy.deepcopy(py_01a()[0]))
    sub_reports.append(copy.deepcopy(py_01b()[0]))
    sub_reports.append(copy.deepcopy(py_01c()[0]))
    sub_reports.append(copy.deepcopy(py_01d()[0]))
    sub_reports.append(copy.deepcopy(py_01e()[0]))
    sub_reports.append(copy.deepcopy(py_01f()[0]))
    sub_reports.append(copy.deepcopy(py_02a()[0]))
    sub_reports.append(copy.deepcopy(py_02b()[0]))
    sub_reports.append(copy.deepcopy(py_02c()[0]))
    sub_reports.append(copy.deepcopy(py_02e()))
    sub_reports.append(copy.deepcopy(py_02e_rtf()))
    sub_reports.append(copy.deepcopy(py_02f()))
    sub_reports.append(copy.deepcopy(py_03a()))
    sub_reports.append(copy.deepcopy(py_03b()))
    sub_reports.append(copy.deepcopy(py_03c()))
    sub_reports.append(copy.deepcopy(py_04b()))

    for report in sub_reports:
        report.collapse(
            [SummaryKey.CORPUS.value], best_selector_key=SummaryKey.ERR.value
        )  # Keep one row for each recognition corpus
        summary_report.merge_report(report, update_structure=True)

    summary_report.set_col_sort_key([SummaryKey.ERR.value, SummaryKey.CORPUS.value])

    tk.register_report("summary.report", summary_report)

    return summary_report
