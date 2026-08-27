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


def multi_scale_cycle_report_format(report: _Report_Type) -> str:
    """
    Report template for the multi-scale RASR sweep. Aggregates the tuning-set WER per
    (lm_scale, prior_scale) combination across the memristor cycles and lists
    mean / std / min / max / n, sorted ascending by mean WER (best first).

    Runs inside GenerateReportStringJob (a running job), so it may resolve sisyphus values.
    """
    import re
    from collections import defaultdict

    report = copy.deepcopy(report)
    instanciate_delayed(report)

    pattern = re.compile(r"search_lm([\d.]+)_prior([\d.]+)")
    groups = defaultdict(list)
    for key, value in report.items():
        if value is None:
            continue
        match = pattern.search(key)
        if match is None:
            continue
        lm, prior = match.group(1), match.group(2)
        try:
            groups[(lm, prior)].append(float(value))
        except (TypeError, ValueError):
            continue

    rows = []
    for (lm, prior), vals in groups.items():
        arr = np.array(vals, dtype="float64")
        rows.append(
            {
                "lm": lm,
                "prior": prior,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(vals),
            }
        )
    rows.sort(key=lambda r: r["mean"])

    lines = ["Multi-scale sweep (short dev), sorted by mean WER (best first):", ""]
    for r in rows:
        lines.append(
            "lm%s/prior%s:  mean %.2f  std %.2f  min %.2f  max %.2f  n %d"
            % (r["lm"], r["prior"], r["mean"], r["std"], r["min"], r["max"], r["n"])
        )
    if not rows:
        lines.append("(no results found)")
    return "\n".join(lines)


def build_memristor_base_report(report: Dict):
    from math import ceil

    report = copy.deepcopy(report)
    baselines = report.pop("baselines")
    best_baselines = {}
    for exp, dic in baselines.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_baselines[exp] = (dic[best], best)
        else:
            best_baselines[exp] = ("None", "")
    line = []
    best_dc = {}
    bits = {1.5, 2, 3, 4, 5, 6, 7, 8}
    line.append("Baselines")
    for exp, best in best_baselines.items():
        line.append(f"{exp.split('/')[4]}: {best[0]}   {' '.join(best[1].split('/')[5:])}")
    line.append("")
    for exp, dic in report.items():
        for bit in bits:
            tmp = {}
            for name in dic:
                # print(bit, name)
                # print(f"weight_{bit}" in name)
                # print(bit == ceil(bit))
                # print(f"weight_{int(bit)}" in name)
                if f"weight_{bit}" in name or (bit == ceil(bit) and f"weight_{int(bit)}" in name):
                    # print(bit, name)
                    tmp[name] = dic[name]
            instanciate_delayed(tmp)
            if all(tmp.values()) and len(tmp) > 0:
                best = min(tmp, key=tmp.get)
                best_dc["/".join(best.split("/")[:9])] = (tmp[best], best)
            else:
                best_dc[exp] = ("None", "")
    for exp, value in best_dc.items():
        if isinstance(exp, float):
            line.append(f"{exp}: {value}")
        else:
            line.append(f"{' '.join(exp.split('/')[4:])}: {value[0]}   {' '.join(value[1].split('/')[9:])}")
    return "\n".join(line)
