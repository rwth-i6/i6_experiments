import copy
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import SummaryKey
from sisyphus import tk

from .config_01a_conformer_ctc_phon import py as py_01a
from .config_01b_conformer_ctc_bpe import py as py_01b
from .config_01c_conformer_ctc_phon_rasrfsa import py as py_01c
from .config_01d_conformer_ctc_bpe_tuning import py as py_01d


def main() -> SummaryReport:
    summary_report = SummaryReport()

    for subreport in [
        copy.deepcopy(py_01a()),
        copy.deepcopy(py_01b()),
        copy.deepcopy(py_01c()),
        copy.deepcopy(py_01d()),
    ]:
        subreport.collapse([SummaryKey.CORPUS.value], best_selector_key=SummaryKey.ERR.value)
        summary_report.merge_report(subreport, update_structure=True)

    summary_report.set_col_sort_key([SummaryKey.ERR.value, SummaryKey.WER.value, SummaryKey.CORPUS.value])

    tk.register_report("summary.report", summary_report)

    return summary_report
