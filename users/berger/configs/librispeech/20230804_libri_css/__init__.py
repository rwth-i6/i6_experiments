from sisyphus import tk

from i6_experiments.users.berger.recipe.summary import SummaryReport
from .config_01_conformer_hybrid_tfgridnet_v2 import py as py_01
from .config_02_conformer_hybrid_blstm_v2 import py as py_02


def main() -> SummaryReport:
    summary_report = SummaryReport()

    summary_report.merge_report(py_01(), update_structure=True, collapse_rows=False)
    summary_report.merge_report(py_02(), collapse_rows=False)

    tk.register_report("summary.report", summary_report)

    return summary_report
