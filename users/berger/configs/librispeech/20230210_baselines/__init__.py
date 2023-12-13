from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

# from i6_experiments.common.baselines.librispeech.ls960.gmm import (
#     baseline_config as gmm_config,
# )
from .config_01a_ctc_blstm_raw_samples import py as py_01a
from .config_01b_ctc_blstm_rasr_features import py as py_01b
from .config_01c_ctc_conformer_raw_samples import py as py_01c
from .config_01d_ctc_conformer_rasr_features import py as py_01d
from .config_02a_transducer_raw_samples import py as py_02a
from .config_02b_transducer_rasr_features import py as py_02b
from .config_03a_transducer_fullsum_raw_samples import py as py_03a
from .config_03b_transducer_fullsum_rasr_features import py as py_03b


def main() -> SummaryReport:
    # gmm_config.run_librispeech_960_common_baseline()

    summary_report = SummaryReport()

    summary_report.merge_report(py_01a()[0], update_structure=True, collapse_rows=True)
    summary_report.merge_report(py_01b()[0], collapse_rows=True)
    summary_report.merge_report(py_01c()[0], collapse_rows=True)
    summary_report.merge_report(py_01d()[0], collapse_rows=True)
    summary_report.merge_report(py_02a()[0], collapse_rows=True)
    summary_report.merge_report(py_02b()[0], collapse_rows=True)
    summary_report.merge_report(py_03a(), collapse_rows=True)
    summary_report.merge_report(py_03b(), collapse_rows=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
