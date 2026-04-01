from sisyphus import tk
import numpy as np
from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type
import copy
from typing import Any, Dict, List
from i6_core.util import instanciate_delayed


def calc_stat(ls):
    avrg = np.average([float(x[1]) for x in ls])
    min = np.min([float(x[1]) for x in ls])
    max = np.max([float(x[1]) for x in ls])
    median = np.median([float(x[1]) for x in ls])
    std = np.std([float(x[1]) for x in ls])
    ex_str = f"Avrg: {avrg}, Min {min}, Max {max}, Median {median}, Std {std},    ({avrg},{min},{max},{median},{std}) Num Values: {len(ls)}"
    return ex_str


def baseline_report_format(report: _Report_Type) -> str:
    """
    Example report format for the baseline , extra ls can be set in order to filter out certain results
    :param report:
    :return:
    """
    extra_ls = ["quantize_static"]
    sets = set()
    for recog in report:
        sets.add(recog.split("/")[-1])
    out = [
        (" ".join(recog.split("/")[3:]), str(report[recog]))
        for recog in report
        if not any(extra in recog for extra in extra_ls)
    ]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = [out[0]]
    if any("cycle" in x[0] for x in best_ls):
        ex_str = calc_stat(out)
        out.insert(0, ("Cycle Statistics: ", ex_str))
    for dataset in sets:
        for extra in extra_ls:
            if extra == "quantize_static":
                tmp = {recog: report[recog] for recog in report if extra in recog and dataset in recog}
                iters = set()
                for recog in tmp:
                    x = recog.split("/")
                    for sub in x:
                        if "samples" in sub:
                            iters.add(sub[len("samples_") :])
                for samples in iters:
                    out2 = [
                        (" ".join(recog.split("/")[3:]), str(report[recog]))
                        for recog in report
                        if f"samples_{samples}/" in recog and dataset in recog
                    ]
                    out2 = sorted(out2, key=lambda x: float(x[1]))
                    if len(out2) > 0:
                        ex_str = calc_stat(out2)
                        out.append(("", ""))
                        out.append((dataset + " " + extra + f"_samples_{samples}", ex_str))
                        # out.extend(out2[:3])
                        # out.extend(out2[-3:])
                        out.extend(out2)
                        best_ls.append(out2[0])
            else:
                out2 = [
                    (" ".join(recog.split("/")[3:]), str(report[recog]))
                    for recog in report
                    if extra in recog and dataset in recog
                ]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    out.append(("", ""))
                    out.append((dataset + " " + extra, ""))
                    out.extend(out2)
                    best_ls.append(out2[0])
    best_ls = sorted(best_ls, key=lambda x: float(x[1]))
    best_ls += [("Base Results", "")]
    out = best_ls + out
    out.insert(0, ("Best Results", ""))
    return "\n".join([f"{pair[0]}:  {str(pair[1])}" for pair in out])


def generate_report(results, exp_name, report_template=baseline_report_format):
    report = GenerateReportStringJob(report_values=results, report_template=report_template)
    report.add_alias(f"report/report/{exp_name}")
    mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
    mail.add_alias(f"report/mail/{exp_name}")
    tk.register_output("mail/" + exp_name, mail.out_status)


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
    dev_results = report_values["dev_results"]
    best_test = report_values["best_test"]

    from i6_core.util import instanciate_delayed

    dev_results = instanciate_delayed(dev_results)
    tuning_tuples = instanciate_delayed(tuning_tuples)

    best_dev = np.argmin(dev_results)

    if tuning_tuples is None:
        assert tuning_names is None
        best_param_str = ""
    else:
        assert len(tuning_tuples[0]) == len(tuning_names)
        best_param_str = "Best values:\n"
        for i, name in enumerate(tuning_names):
            best_param_str += f"{name}: {tuning_tuples[best_dev][i]}\n"
        best_param_str += "\n"

    name_str = training_name + "\n\n"

    final_results_str = f"""
Final results:
dev: {dev_results[best_dev]}
tes: {best_test.get()}   
"""

    latex_str = f"Latex:\n{dev_results[best_dev]} & {best_test.get()}\\\\ \n\n"

    return name_str + best_param_str + final_results_str + latex_str


def tune_and_evalue_report(
    training_name: str,
    tune_parameters: List[Any],
    tuning_names: List[str],
    tune_values_dev: List[tk.Variable],
    report_values: Dict[str, tk.Variable],
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
        "dev_results": tune_values_dev,
        "best_test": report_values["test"],
    }
    report = GenerateReportStringJob(
        report_values=report_values, report_template=search_report, compress=False
    ).out_report
    tk.register_output(training_name + "/tune_and_evaluate_report.txt", report)