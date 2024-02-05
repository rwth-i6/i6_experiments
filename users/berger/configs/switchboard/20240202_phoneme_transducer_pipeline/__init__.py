from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from .config_01a_ctc_blstm import py as py_01a
from .config_01b_ctc_conformer import py as py_01b


def main() -> SummaryReport:
    # gmm_config.run_librispeech_960_common_baseline()

    summary_report = SummaryReport()

    summary_report.merge_report(py_01a()[0], update_structure=True, collapse_rows=True)
    summary_report.merge_report(py_01b()[0], collapse_rows=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
