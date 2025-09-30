from sisyphus import tk, Job, Task
from typing import Dict, Iterable, List, Iterator
class ReportDictJob(Job):
    """
    reports.
    """
    def __init__(
        self,
        *,
        outputs: Dict[str, tk.Path | tk.Variable],
    ):
        super(ReportDictJob, self).__init__()
        self.outputs = outputs  # type: Dict[str, tk.Path]
        self.out_report_dict = self.output_path("report.py")

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", rqmt={"cpu":1, "time":1})#mini_task=True)

    def run(self):
        """run"""
        with open(self.out_report_dict.get_path(), "wt") as out:
            out.write("{\n")
            for name, res in self.outputs.items():
                if isinstance(res, tk.Variable):
                    res = res.get()
                elif isinstance(res, tk.Path):
                    with open(res, "rt") as infile:
                        res = infile.read()
                out.write(f"\t{name!r}: \n\t'{res}',\n")
            out.write("}\n")

from i6_experiments.users.zeyer.datasets.score_results import ScoreResult
PRINTED = False
class GetOutPutsJob(Job):
    """
    Collect all wer reports from recogs.
    """
    def __init__(
        self,
        *,
        outputs: Dict[str, Dict[str, ScoreResult]],
    ):
        """
        :param model: modelwithcheckpoints, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        """
        super(GetOutPutsJob, self).__init__()
        global PRINTED
        self.outputs = outputs  # type: Dict[str, Dict[str, ScoreResult]]
        self.out_report_dict = self.output_path("wer_report.py")
        if not PRINTED:
            for k,v in self.outputs.items():
                for dataset, report in v.items():
                    print(f"{k}:\n{dataset}: {report}")
                    break
            #print(self.outputs)
            PRINTED = True

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", rqmt={"cpu":1, "time":1})#mini_task=True)

    def run(self):
        """run"""
        with open(self.out_report_dict.get_path(), "wt") as out:
            out.write("{\n")
            for lm, wer_dict in self.outputs.items():
                out.write(f"\t{lm!r}" +":{\n")
                for dataset, score_res in wer_dict.items():
                    out.write(f"{dataset!r}: {score_res.report}\n")
                out.write("\t}\n")
            out.write("}\n")