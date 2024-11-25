import copy

from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey
from sisyphus import tk

from .config_01_conformer_ctc import py as py_01
from .config_01a_conformer_ctc_ogg import py as py_01a


def main() -> SummaryReport:
    summary_report = SummaryReport()

    for subreport in [
        copy.deepcopy(py_01()[0]),
        copy.deepcopy(py_01a()[0]),
    ]:
        subreport.collapse([SummaryKey.CORPUS.value], best_selector_key=SummaryKey.ERR.value)
        summary_report.merge_report(subreport, update_structure=True)

    summary_report.set_col_sort_key([SummaryKey.ERR.value, SummaryKey.WER.value, SummaryKey.CORPUS.value])

    tk.register_report("summary.report", summary_report)

    return summary_report
