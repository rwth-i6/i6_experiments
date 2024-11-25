from functools import partial
import numpy as np
from typing import Any, Dict, List

from sisyphus import tk

from i6_core.report.report import GenerateReportStringJob


def search_report(report_values):
    """
    This function is called from within the report, so as a running job
    Thus, is allowed to resolve sisyphus values

    Reports for flashlight lexical search with different tuning

    :param training_name: some name to print
    :param tuning_tuples: list of tuples, e.g. [(lm_scale, prior_scale), ...]
    :param tuning_names: list of string with the length of the tuples, giving a description of the parameter
    :param dev_clean_results: dev clean values with length of tuning
    :param dev_other_results:  same as above
    :param best_test_clean: single best test clean result
    :param best_test_other: single best test other result
    :return: 
    """
    training_name = report_values["training_name"]
    tuning_tuples = report_values["tuning_tuples"]
    tuning_names = report_values["tuning_names"]
    dev_clean_results = report_values["dev_clean_results"]
    dev_other_results = report_values["dev_other_results"]
    best_test_clean = report_values["best_test_clean"]
    best_test_other = report_values["best_test_other"]

    from i6_core.util import instanciate_delayed
    dev_clean_results = instanciate_delayed(dev_clean_results)
    dev_other_results = instanciate_delayed(dev_other_results)
    tuning_tuples = instanciate_delayed(tuning_tuples)

    best_clean = np.argmin(dev_clean_results)
    best_other = np.argmin(dev_other_results)

    if tuning_tuples is None:
        assert tuning_names is None
        best_param_clean_str = ""
        best_param_other_str = ""
    else:
        assert len(tuning_tuples[0]) == len(tuning_names)
        best_param_clean_str = "Best values clean:\n"
        best_param_other_str = "Best values other:\n"
        for i, name in enumerate(tuning_names):
            best_param_clean_str += f"{name}: {tuning_tuples[best_clean][i]}\n"
            best_param_other_str += f"{name}: {tuning_tuples[best_other][i]}\n"
        best_param_clean_str += "\n"
        best_param_other_str += "\n"

    name_str = training_name + "\n\n"

    final_results_str = f"""
Final results:
dev-clean: {dev_clean_results[best_clean]}
dev-other: {dev_other_results[best_other]}
test-clean: {best_test_clean.get()}
test-clean: {best_test_other.get()}   
"""

    latex_str = f"Latex:\n{dev_clean_results[best_clean]} & {dev_other_results[best_other]} & {best_test_clean.get()} & {best_test_other.get()}\\\\ \n\n"

    return name_str + best_param_clean_str + best_param_other_str + final_results_str + latex_str

def tune_and_evalue_report(
        training_name: str,
        tune_parameters: List[Any],
        tuning_names: List[str],
        tune_values_clean: List[tk.Variable],
        tune_values_other: List[tk.Variable],
        report_values: Dict[str, tk.Variable]
):
    """
    A helper function for the reporting, specifically targeting the tune_and_evalaute_helper for lexical search

    :param training_name: needs to be unique, is used for hash
    :param tune_parameters: list of tuples containing tunable parameters
    :param tuning_names: list of names for each tunable parameter
    :param tune_values_clean: dev-clean WERs, same length as list of tuples
    :param tune_values_other: dev-other WERs,
    :param report_values: report value dict containing "test-clean" and "test-other" entry
    :return:
    """
    report_values = {
        "training_name": training_name,
        "tuning_tuples": tune_parameters,
        "tuning_names": tuning_names,
        "dev_clean_results": tune_values_clean,
        "dev_other_results": tune_values_other,
        "best_test_clean": report_values["test-clean"],
        "best_test_other": report_values["test-other"],
    }
    report = GenerateReportStringJob(report_values=report_values, report_template=search_report,
                                     compress=False).out_report
    tk.register_output(training_name + "/tune_and_evaluate_report.txt", report)


