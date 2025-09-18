from typing import List, Optional, Dict, Any, List, Union, Tuple
from i6_core.util import instanciate_delayed


def build_base_report(report: Dict):
    best_dc = {}
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
        line.append(f"{' '.join(exp.split('.')[2:])}: {value[0]}   {' '.join(value[1].split('/')[7:])}")
    return "\n".join(line)
