from typing import Dict

from i6_core.report.report import GenerateReportStringJob, _Report_Type
from i6_core.util import instanciate_delayed

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


def create_generate_report_job(results, exp_name: str, report_template=baseline_report_format) -> None:
    report_job = GenerateReportStringJob(report_values=results, report_template=report_template)
    report_job.add_alias(f"report/report/{exp_name}")


def build_base_report(report: Dict):
    best_dc = {}
    for exp, dic in report.items():
        instanciate_delayed(dic)
        new_dic = {k: v for k, v in dic.items() if "other" in k}
        if all(new_dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(new_dic[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test-other" in dic:
                best_dc["/".join(best.split("/")[:-2]) + "/test-other"] = (
                    "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test-other"])),
                    best,
                )
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    return "\n".join(line)