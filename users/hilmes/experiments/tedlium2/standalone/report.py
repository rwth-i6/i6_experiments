from sisyphus import tk
import numpy as np
from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type
import copy
from typing import Dict
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


def build_memristor_base_report(report: Dict):
    report = copy.deepcopy(report)
    bits = set()
    instanciate_delayed(report)
    best_dc = {}
    base = {}
    from math import ceil

    for exp in report:
        if not "mem" in exp:
            base[exp] = report[exp]
        else:
            pref = exp.split("_")[12]
            bits.add(float(pref))
    for bit in bits:
        dc = {}
        for exp in report:
            if f"weight_{bit}" in exp or (bit == ceil(bit) and f"weight_{int(bit)}" in exp):
                dc[exp] = report[exp]
        if all(dc.values()):
            best = min(dict(dc), key=dc.get)
            best_dc[best] = ("{:.1f}".format(float(dc[best])), best)
        else:
            best_dc[bit] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        line.append(f"{exp.split('/')[6].split('_')[0]}: {value[0]}   {' '.join(value[1].split('/')[7:])}")
    return "\n".join(line)
