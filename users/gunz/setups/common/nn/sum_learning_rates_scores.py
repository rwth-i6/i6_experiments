import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterator, List, Union

from sisyphus import tk, Job, Path, Task


class SumScoresInLearningRatesFileJob(Job):
    def __init__(self, lr_file: Path, keys: Union[tk.Variable, List[str]], out_key: str = "sum_score"):
        self.lr_file = lr_file
        self.keys = keys
        self.out_key = out_key

        self.out_learning_rates = self.output_path("learning_rates")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        @dataclass
        class EpochData:
            learningRate: float
            error: Dict[str, float]

        with open(self.lr_file, "rt") as f:
            data: Dict[int, EpochData] = eval(
                f.read(),
                {"nan": float("nan"), "inf": float("inf"), "EpochData": EpochData},
            )

        keys = self.keys if isinstance(self.keys, list) else self.keys.get()
        assert len(keys) > 0

        out_lr = {
            epoch: dataclasses.replace(
                vals,
                error={
                    **vals.error,
                    self.out_key: sum((vals.error[k] for k in keys)),
                },
            )
            for epoch, vals in data.items()
        }
        with open(self.out_learning_rates, "wt") as f:
            f.write(repr(out_lr))
