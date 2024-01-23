import os
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from sisyphus import tk

from .config_00_gmm import py as py_00_gmm
from .config_01_blstm_hybrid import py as py_01_hybrid
from .config_02_blstm_ctc import py as py_02_ctc
from .config_03_blstm_transducer import py as py_03_transducer
from .config_04_blstm_ctc_bpe import py as py_04_ctc_bpe
from .config_05_conformer_hybrid import py as py_05_conf_hybrid

dir_handle = os.path.dirname(__file__).split("config/")[1]


def py() -> SummaryReport:
    summary_report = SummaryReport()
    gmm_outputs = py_00_gmm()

    gmm_alignments = {key: output.alignments for key, output in gmm_outputs.items()}
    cart_file = gmm_outputs["train_si284"].crp.acoustic_model_config.state_tying.file

    summary_report.merge_report(
        py_01_hybrid(gmm_alignments, cart_file),
        update_structure=True,
        collapse_rows=True,
    )
    ctc_alignments, summary = py_02_ctc()
    summary_report.merge_report(summary, collapse_rows=True)
    summary_report.merge_report(py_03_transducer(ctc_alignments), collapse_rows=True)
    summary_report.merge_report(py_04_ctc_bpe(), collapse_rows=True)
    summary_report.merge_report(py_05_conf_hybrid(gmm_alignments, cart_file), collapse_rows=True)

    tk.register_report(
        f"{dir_handle}/summary.report",
        summary_report,
    )

    return summary_report
