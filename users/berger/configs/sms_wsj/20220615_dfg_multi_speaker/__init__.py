import os

from sisyphus import tk

from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from .wsj_8kHz.config_main import py as py_wsj_8kHz
from .wsj_16kHz.config_main import py as py_wsj_16kHz
from .sms_wsj_8kHz.config_main import py as py_sms_wsj_8kHz
from .sms_wsj_16kHz.config_main import py as py_sms_wsj_16kHz


def main() -> SummaryReport:
    summary_report = SummaryReport()

    summary = py_wsj_8kHz()
    summary.add_column("Dataset", 1, "wsj_8kHz")
    summary_report.merge_report(summary, update_structure=True, collapse_rows=False)
    summary = py_wsj_16kHz()
    summary.add_column("Dataset", 1, "wsj_16kHz")
    summary_report.merge_report(summary, collapse_rows=False)
    summary = py_sms_wsj_8kHz()
    summary.add_column("Dataset", 1, "sms_wsj_8kHz")
    summary_report.merge_report(summary, collapse_rows=False)
    summary = py_sms_wsj_16kHz()
    summary.add_column("Dataset", 1, "sms_wsj_16kHz")
    summary_report.merge_report(summary, collapse_rows=False)

    tk.register_report(
        "summary.report",
        summary_report,
    )
    return summary_report
