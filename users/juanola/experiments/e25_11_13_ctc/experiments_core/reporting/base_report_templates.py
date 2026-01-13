from typing import Dict

from i6_core.report.report import _Report_Type
from i6_core.util import instanciate_delayed


def base_report_template_v0(results_per_experiment: Dict[str, _Report_Type]) -> str:
    """
    Generates a report for results of different experiments
    :param results_per_experiment:
    :return:
    """
    best_dc = {}

    for exp_name, exp_results in results_per_experiment.items():
        instanciate_delayed(exp_results)
        new_dic = {k: v for k, v in exp_results.items() if "other" in k}
        if all(new_dic.values()):
            best = min(new_dic, key=new_dic.get)
            best_dc[" ".join(exp_name.split("/")[5:])] = ("{:.1f}".format(float(new_dic[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test-other" in exp_results:
                best_dc["/".join(best.split("/")[:-2]) + "/test-other"] = (
                    "{:.1f}".format(float(exp_results["/".join(best.split("/")[:-2]) + "/test-other"])),
                    best,
                )
        else:
            best_dc[" ".join(exp_name.split("/")[5:])] = ("None", "")

    line = []
    for exp_name, value in best_dc.items():
        line.append(f"{' '.join(exp_name.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[6:])}")
    return "\n".join(line)