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