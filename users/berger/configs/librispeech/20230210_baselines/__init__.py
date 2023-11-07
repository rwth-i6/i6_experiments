from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

# from i6_experiments.common.baselines.librispeech.ls960.gmm import (
#     baseline_config as gmm_config,
# )
from .config_01a_ctc_blstm import py as py_01a
from .config_01b_ctc_conformer import py as py_01b
from .config_01c_ctc_conformer_rasr_features import py as py_01c
from .config_02_transducer import py as py_02
from .config_03_transducer_fullsum import py as py_03
from .config_02a_transducer_rasr_features import py as py_02a
from .config_03a_transducer_fullsum_rasr_features import py as py_03a


def main() -> SummaryReport:
    # gmm_config.run_librispeech_960_common_baseline()

    summary_report = SummaryReport()

    summary_report.merge_report(py_01a()[0], update_structure=True, collapse_rows=True)
    summary_report.merge_report(py_01b()[0], collapse_rows=True)
    summary_report.merge_report(py_01c()[0], collapse_rows=True)
    summary_report.merge_report(py_02()[0], collapse_rows=True)
    summary_report.merge_report(py_03(), collapse_rows=True)
    summary_report.merge_report(py_02a()[0], collapse_rows=True)
    summary_report.merge_report(py_03a(), collapse_rows=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
