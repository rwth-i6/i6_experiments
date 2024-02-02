from sisyphus import tk
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

# from i6_experiments.common.baselines.librispeech.ls960.gmm import (
#     baseline_config as gmm_config,
# )
from .config_01a_ctc_blstm_raw_samples import py as py_01a
from .config_01b_ctc_blstm_rasr_features import py as py_01b
from .config_01c_ctc_conformer_raw_samples import py as py_01c
from .config_01d_ctc_conformer_rasr_features import py as py_01d
from .config_01e_ctc_conformer_rasr_features_dc import py as py_01e
from .config_02a_transducer_raw_samples import py as py_02a
from .config_02b_transducer_rasr_features import py as py_02b

# from .config_02c_transducer_wei import py as py_02c
# from .config_02d_transducer_rasr_features_dc import py as py_02d
from .config_03a_transducer_fullsum_raw_samples import py as py_03a
from .config_03b_transducer_fullsum_rasr_features import py as py_03b

# from .config_03c_transducer_fullsum_wei import py as py_03c

# from .config_test_1 import py as py_test_1
# from .config_test_2 import py as py_test_2
# from .config_test_3 import py as py_test_3
# from .config_test_4 import py as py_test_4
# from .config_test_5 import py as py_test_5
# from .config_test_6 import py as py_test_6
# from .config_test_7 import py as py_test_7
# from .config_test_8 import py as py_test_8


def main() -> SummaryReport:
    # gmm_config.run_librispeech_960_common_baseline()

    summary_report = SummaryReport()

    summary_report.merge_report(py_01a()[0], update_structure=True, collapse_rows=True)
    summary_report.merge_report(py_01b()[0], collapse_rows=True)
    summary_report.merge_report(py_01c()[0], collapse_rows=True)
    summary_report.merge_report(py_01d()[0], collapse_rows=True)
    summary_report.merge_report(py_01e()[0], collapse_rows=True)
    summary_report.merge_report(py_02a()[0], collapse_rows=True)
    summary_report.merge_report(py_02b()[0], collapse_rows=True)
    # summary_report.merge_report(py_02c()[0], collapse_rows=True)
    # summary_report.merge_report(py_02d()[0], collapse_rows=True)
    summary_report.merge_report(py_03a(), collapse_rows=True)
    summary_report.merge_report(py_03b(), collapse_rows=True)
    # summary_report.merge_report(py_03c(), collapse_rows=True)

    # summary_report.merge_report(py_test_1()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_2()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_3()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_4()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_5()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_6()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_7()[0], collapse_rows=True)
    # summary_report.merge_report(py_test_8()[0], collapse_rows=True)

    tk.register_report("summary.report", summary_report)

    return summary_report
