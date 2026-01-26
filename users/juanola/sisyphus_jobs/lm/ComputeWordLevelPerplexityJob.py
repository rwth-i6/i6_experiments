from typing import Union, Optional

from sisyphus import *


class ComputeWordLevelPerplexityJob(Job):
    def __init__(self, sw_ppl_file: Optional[Path], exponent: Union[float, tk.Variable],  precision_ndigit=1):
        self.sw_ppl_file = sw_ppl_file
        self.exponent = exponent
        self.precision_ndigit = precision_ndigit

        self.out_ppl = self.output_path("wl_ppl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):
        import math

        fpath = self.sw_ppl_file.get_path()
        if fpath.endswith(".gz"):
            import gzip
            open_func = gzip.open
        else:
            open_func = open

        with open_func(fpath, "rt") as f:
            sw_ppl = float(f.read())
        exponent = self.exponent.get() if isinstance(self.exponent, tk.Variable) else self.exponent

        wl_ppl = math.pow(sw_ppl, exponent)
        if self.precision_ndigit is not None:
            wl_ppl = round(wl_ppl, self.precision_ndigit)

        with open(self.out_ppl.get_path(), "w+") as f:
            f.write(f"{wl_ppl}")
