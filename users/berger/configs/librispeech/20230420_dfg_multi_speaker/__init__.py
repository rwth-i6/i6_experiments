from sisyphus import tk

from i6_experiments.users.berger.recipe.summary import SummaryReport
from .config_01_blstm_hybrid_mixed_inputs import py as py_01
from .config_02_blstm_hybrid_mixed_inputs_joint import py as py_02


def main() -> SummaryReport:
    summary_report = SummaryReport()

    summary_report.merge_report(py_01()[0], update_structure=True, collapse_rows=True)
    summary_report.merge_report(py_02(), collapse_rows=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
