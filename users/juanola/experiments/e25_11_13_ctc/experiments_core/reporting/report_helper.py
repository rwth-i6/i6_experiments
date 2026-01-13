from typing import Dict, Any

from i6_core.report.report import GenerateReportStringJob
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.constants import (
    SIS_OUTPUTS_EXP_REPORTS,
    SIS_ALIASES_REPORTS,
)
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.experiments_core.reporting.templates.experiment_report_templates import (
    experiment_report_template_v0,
)
from sisyphus import tk


def generate_experiment_results_report(
    exp_results: Dict[str, Any], exp_name: str, report_template=experiment_report_template_v0
) -> None:
    """
    Generates a report for the results of one experiment.
    Also adds an alias for the reporting job.
    :param exp_results:
    :param exp_name:
    :param report_template:
    :return:
    """
    report_job = GenerateReportStringJob(report_values=exp_results, report_template=report_template, compress=False)
    tk.register_output(f"{SIS_OUTPUTS_EXP_REPORTS}/{exp_name}", report_job.out_report)
    report_job.add_alias(f"{SIS_ALIASES_REPORTS}/{exp_name}")
