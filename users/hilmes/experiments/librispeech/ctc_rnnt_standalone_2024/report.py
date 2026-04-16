from functools import partial
import numpy as np
from typing import Any, Dict, List

from sisyphus import tk

import copy
from i6_core.util import instanciate_delayed

from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type


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
        "dev_clean_results": tune_values_clean,
        "dev_other_results": tune_values_other,
        "best_test_clean": report_values["test-clean"],
        "best_test_other": report_values["test-other"],
    }
    report = GenerateReportStringJob(
        report_values=report_values, report_template=search_report, compress=False
    ).out_report
    tk.register_output(training_name + "/tune_and_evaluate_report.txt", report)


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
        if not any(extra in recog for extra in extra_ls) and "clean" not in recog
    ]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = [out[0]]
    for dataset in sets:
        for extra in extra_ls:
            if extra == "":
                continue
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


# def build_memristor_base_report(report: Dict):
    # from math import ceil
    #
    # report = copy.deepcopy(report)
    # baselines = report.pop("baselines")
    # best_baselines = {}
    # for exp, dic in baselines.items():
    #     instanciate_delayed(dic)
    #     new_dic = {k: v for k, v in dic.items() if "other" in k}
    #     if all(new_dic.values()):
    #         best = min(new_dic, key=new_dic.get)
    #         best_baselines[exp + "/" + best] = new_dic[best]
    #     else:
    #         best_baselines[exp] = "None"
    # line = []
    # best_dc = {}
    # bits = {1.5, 2, 3, 4, 5, 6, 8}
    # for exp, best in best_baselines.items():
    #     line.append(f"{exp.split('/')[5]}: {best}   {' '.join(exp.split('/')[10:])}")
    # for exp, dic in report.items():
    #     tmp = {}
    #     for bit in bits:
    #         for name in dic:
    #             if f"weight_{bit}" in name or (bit == ceil(bit) and f"weight_{int(bit)}" in name):
    #                 tmp[name] = dic[name]
    #         if all(tmp.values()):
    #             instanciate_delayed(tmp)
    #             best = min(tmp, key=tmp.get)
    #             best_dc[exp + "/" + best] = tmp[best]
    #         else:
    #             best_dc[exp] = "None"
    # for exp, value in best_dc.items():
    #     if isinstance(exp, float):
    #         line.append(f"{exp}: {value}")
    #     else:
    #         line.append(f"{' '.join(exp.split('/')[9:12])}: {value}   {' '.join(exp.split('/')[12:])}")
    # return "\n".join(line)

def build_qat_report(report: Dict):
    import numpy as np

    exps = ["combined", "cycle", "smaller", "greedy"]
    report = copy.deepcopy(report)
    best_dc = {}
    bits = [8, 7, 6, 5, 4, 3, 2, 1.5]
    for exp, dic in report.items():
        instanciate_delayed(dic)
        new_dic = {k: v for k, v in dic.items() if "other" in k}
        if all(new_dic.values()):
            if len(new_dic) == 0:
                print(exp)
                assert False
            best = min(new_dic, key=new_dic.get)
            if "cycle" in exp:
                mean = np.round(np.mean(list(new_dic.values())), decimals=2)
                mini = np.min(list(new_dic.values()))
                maxi = np.max(list(new_dic.values()))
                std = np.round(np.std(list(new_dic.values())), decimals=4)
                ln = len(new_dic.values())
                best_dc[" ".join(exp.split("/")[3:])] = (
                    "{:.1f}".format(mean),
                    best + f"/{mean=} {std=} min: {'{:.1f}'.format(mini)} max: {'{:.1f}'.format(maxi)} runs: {ln}",
                )
            else:
                best_dc[" ".join(exp.split("/")[3:])] = ("{:.1f}".format(float(new_dic[best])), best)
        else:
            best_dc[" ".join(exp.split("/")[3:])] = ("None", "")
    line = []
    tmp = copy.deepcopy(best_dc)
    for exp, value in best_dc.items():
        if "baseline" in exp:
            line.append(
                f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[5:6] + value[1].split('/')[9:])}")
            del tmp[exp]
    best_dc = tmp
    tmp = copy.deepcopy(best_dc)
    for bit in bits:
        best_dc = tmp
        tmp = copy.deepcopy(best_dc)
        first = False
        for exp, value in best_dc.items():
            if any(x in exp for x in exps):
                continue
            if f"{bit}_8" not in exp:
                continue
            line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
            if first == False:
                first = True
            del tmp[exp]
        if first == True:
            line.append("")
    tmp = best_dc
    for x in exps:
        first = True
        best_dc = copy.deepcopy(tmp)
        if x == "noise":
            w_bits = set()
            a_bits = set()
            starts = set()
            devs = set()
            dropouts = set()
            for exp in best_dc:
                if "noise" in exp:
                    pref = "_".join(exp.split("_")[:3])
                    w_bits.add(exp.split("_")[3])
                    a_bits.add(exp.split("_")[4])
                    starts.add(exp.split("_")[5][len("noise"):])
                    devs.add(exp.split("_")[6])
                    dropouts.add(exp.split("_")[7].split(" ")[0][len("drop"):])
            if all(len(x) == 0 for x in [w_bits, a_bits, starts, devs, dropouts]):
                continue
            line.append(x)
            line.append("Weight B".ljust(10) + "Start".ljust(10) + "Dropout".ljust(10) + "Deviation".ljust(10) + "No Noise".ljust(10) + "Noise".ljust(10) + "Memristor".ljust(10))
            for w_bit in sorted(w_bits, reverse=True):
                for a_bit in sorted(a_bits, reverse=True):
                    for start in sorted(starts):
                        for dev in sorted(devs):
                            for dropout in sorted(dropouts):
                                if pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" without_noise" in best_dc:
                                    no_noise = best_dc[pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" without_noise"]
                                    del tmp[pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" without_noise"]
                                else:
                                    no_noise = "NaN"
                                if pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" with_noise" in best_dc:
                                    noise = best_dc[pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" with_noise"]
                                    del tmp[pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" with_noise"]
                                else:
                                    noise = "NaN"
                                if pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" cycle_combined" in best_dc:
                                    cycle = best_dc[pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" cycle_combined"]
                                    del tmp[pref+"_"+w_bit+"_"+a_bit+"_noise"+start+"_"+dev+"_drop"+dropout+" cycle_combined"]
                                else:
                                    cycle = "NaN"
                                if all(x == "NaN" for x in [no_noise, noise, cycle]):
                                    continue
                                line.append(f"{str(w_bit).ljust(10)}{str(start).ljust(10)}{str(dropout).ljust(10)}{str(dev).ljust(10)}{no_noise[0].ljust(10)}{noise[0].ljust(10)}{cycle[0].ljust(10)}{' '.join(cycle[1].split(' ')[1:]).ljust(25)} {' ' if len(cycle[1]) <= 1 else cycle[1].split('/')[-2]}")
            line.append("")
        elif x == "correction":
            w_bits = set()
            a_bits = set()
            cycles = set()
            devs = set()
            tests = set()
            for exp in best_dc:
                if "correction_" in exp and not "correction_baseline" in exp:
                    pref = "_".join(exp.split("_")[:3])
                    w_bits.add(exp.split("_")[3])
                    a_bits.add(exp.split("_")[4])
                    cycles.add(exp.split("_")[6])
                    devs.add(exp.split("_")[8][:-len(" cycle")])
                    tests.add(exp.split("_")[7])
            if all(len(x) == 0 for x in [w_bits, a_bits, cycles, devs, tests]):
                continue
            line.append(x)
            line.append("Weight B".ljust(10) + "Cycles".ljust(10) + "Test Val".ljust(10) + "Deviation".ljust(10) + "Correction".ljust(10))
            for w_bit in sorted(w_bits, reverse=True):
                for a_bit in sorted(a_bits, reverse=True):
                    line.append(f"{w_bit.ljust(10)}Baseline: {best_dc[pref + '_' + w_bit + '_' + a_bit + '_correction_baseline'][0]}")
                    line.append(
                        f"{w_bit.ljust(10)}No Correction:      {best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][0]} {' '.join(best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][1].split(' ')[1:])} {best_dc[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined'][1].split('/')[-2]}")
                    del tmp[pref + '_' + w_bit + '_' + a_bit + '_correction_baseline']
                    del tmp[pref + '_' + w_bit + '_' + a_bit + '_no_correction cycle_combined']
                    for cycle in sorted(cycles):
                        for dev in sorted(devs):
                            for test_v in sorted(tests):
                                if pref+"_"+w_bit+"_"+a_bit+"_correction_"+cycle+"_"+test_v+"_"+dev+" cycle_combined" in best_dc:
                                    mem = best_dc[pref+"_"+w_bit+"_"+a_bit+"_correction_"+cycle+"_"+test_v+"_"+dev+" cycle_combined"]
                                    del tmp[pref+"_"+w_bit+"_"+a_bit+"_correction_"+cycle+"_"+test_v+"_"+dev+" cycle_combined"]
                                else:
                                    continue
                                line.append(f"{str(w_bit).ljust(10)}{str(cycle).ljust(10)}{str(test_v).ljust(10)}{str(dev).ljust(10)}{mem[0].ljust(10)}{' '.join(mem[1].split(' ')[1:])} {' ' if len(mem[1]) <= 1 else mem[1].split('/')[-2].ljust(25)}")
            line.append("")
        else:
            for exp, value in best_dc.items():
                if x in exp:
                    if first is True:
                        line.append(x)
                        line.append("")
                        first = False
                    line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[5:6] + value[1].split('/')[9:])}")
                    del tmp[exp]
        if first is False:
            line.append("")
    assert len(tmp) == 0, tmp
    return "\n".join(line)
