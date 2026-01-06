import copy
from typing import List, Optional, Dict, Any, List, Union, Tuple
from i6_core.util import instanciate_delayed


def build_base_report(report: Dict, print_larger_params: bool = True):
    best_dc = {}
    report = copy.deepcopy(report)
    for exp, dic in report.items():
        instanciate_delayed(dic)
        tmp = {x: dic[x] for x in dic.keys() if not "test" in x}
        if all(tmp.values()):
            best = min(tmp, key=tmp.get)
            best_dc[" ".join(exp.split("/")[5:])] = ("{:.1f}".format(float(tmp[best])), best)
            if "/".join(best.split("/")[:-2]) + "/test" in dic:
                if dic["/".join(best.split("/")[:-2]) + "/test"] is not None:
                    best_dc["/".join(best.split("/")[:-2]) + "/test"] = (
                        "{:.1f}".format(float(dic["/".join(best.split("/")[:-2]) + "/test"])),
                        best,
                    )
                else:
                    best_dc["/".join(best.split("/")[:-2]) + "/test"] = ("None", "")
        else:
            best_dc[" ".join(exp.split("/")[5:])] = ("None", "")
    line = []
    for exp, value in best_dc.items():
        if any(y in exp for y in ["larger_search", "largerer_search"]):
            continue

        ln = f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[7:])}"
        wers = [value[0]]
        if exp + "_larger_search" in best_dc:
            ln += (
                f";  {best_dc[exp + '_larger_search'][0]} {' '.join(best_dc[exp + '_larger_search'][1].split('/')[7:]) if print_larger_params else ''}"
            )
            wers.append(best_dc[exp + "_larger_search"][0])
        if exp + "_largerer_search" in best_dc:
            if not exp + "_larger_search" in best_dc:
                ln += "    None     "
            ln += f";  {best_dc[exp + '_largerer_search'][0]} {' '.join(best_dc[exp + '_largerer_search'][1].split('/')[7:]) if print_larger_params else ''}"
            wers.append(best_dc[exp + "_largerer_search"][0])
        if len(wers) > 1:
            ln += f"  {', '.join(wers)}"
        line.append(ln)

    return "\n".join(line)
