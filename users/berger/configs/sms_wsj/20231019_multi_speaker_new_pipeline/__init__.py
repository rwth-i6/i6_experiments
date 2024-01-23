from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from .config_01a_ctc_blstm import py as py_01a
from .config_01b_ctc_conformer import py as py_01b
from .config_02a_transducer_blstm import py as py_02a
from .config_02b_transducer_conformer import py as py_02b
from .config_03a_transducer_blstm_fullsum import py as py_03a
from .config_03b_transducer_conformer_fullsum import py as py_03b


def main() -> SummaryReport:
    summary_report = SummaryReport()

    summary_report.merge_report(py_01a()[0], update_structure=True, collapse_rows=True)
    summary_report.merge_report(py_01b()[0], collapse_rows=True)
    summary_report.merge_report(py_02a()[0], collapse_rows=True)
    summary_report.merge_report(py_02b()[0], collapse_rows=True)
    summary_report.merge_report(py_03a(), collapse_rows=True)
    summary_report.merge_report(py_03b(), collapse_rows=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
