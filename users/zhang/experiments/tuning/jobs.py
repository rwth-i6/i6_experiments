import sisyphus
from sisyphus import tk
import json
from typing import Iterator

class SetTuneRangeVars(sisyphus.Job):
    def __init__(
        self,
        tune_values: list[float],
        default_value: [tk.Variable | tk.Path],
    ):
        self.tune_values = tune_values
        self.default_value = default_value
        self.out_tune_values = [self.output_var(f'{i}') for i, _ in enumerate(tune_values)]
    def tasks(self) -> Iterator[sisyphus.Task]:
        """tasks"""
        yield sisyphus.Task("run", rqmt={"cpu": 1, "mem": 1, "time": 1})
    def run(self):
        if isinstance(self.default_value, tk.Path):
            default_value = json.load(open(self.default_value.get_path()))
        elif isinstance(self.default_value, tk.Variable):
            default_value = self.default_value.get()
        else:
            raise TypeError(f"Not known type:{self.default_value} - {type(self.default_value)}")
        for i, value in enumerate(self.tune_values):
            self.out_tune_values[i].set(max(0, value + default_value))