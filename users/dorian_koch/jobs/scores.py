import collections
from typing import Dict, List, Sequence, Tuple, Any
from i6_experiments.users.zeyer.recog import RecogOutput
from sisyphus import Job, Task, tk
from i6_core.util import uopen
import re


class AverageScores(Job):
    def __init__(self, *, scores: RecogOutput):
        self.scores = scores

        self.out_avg = self.output_var("avg")
        self.out_median = self.output_var("median")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        # d["__version"] = 3
        return super().hash(d)

    def tasks(self):
        # yield Task("run", rqmt={"cpu": 16, "mem": 8, "time": 1})
        yield Task("run", mini_task=True)

    def run(self):
        # See TextDictDataset
        with uopen(self.scores.output) as f:
            txt = f.read()
        from returnn.util.literal_py_to_pickle import literal_eval

        # Note: literal_py_to_pickle.literal_eval is quite efficient.
        # However, currently, it does not support inf/nan literals,
        # so it might break for some input.
        # We might want to put a simple fallback to eval here if needed.
        # Or maybe extend literal_py_to_pickle.literal_eval to support inf/nan literals.
        try:
            data: Dict[str, Any] = literal_eval(txt)
        except Exception as exc:
            print(f"{self}: Warning: literal_py_to_pickle.literal_eval failed:")
            print(f"  {type(exc).__name__}: {exc}")
            print("  Fallback to eval...")
            data: Dict[str, Any] = eval(txt)

        assert data is not None
        assert isinstance(data, dict)
        assert len(data) > 0
        # Check some data.
        key, value = next(iter(data.items()))
        assert isinstance(key, str), f"{self}: expected seq tag as keys, got {key!r} ({type(key)})"  # seq tag
        # if self.item_format == "single":
        #     assert isinstance(value, str), f"{self}: expected str ({self.item_format}), got {value!r} ({type(value)})"
        # elif self.item_format == "list_with_scores":
        if True:
            assert isinstance(value, list), f"{self}: expected list ({self.item_format}), got {value!r} ({type(value)})"
            assert len(value) > 0, f"{self}: expected non-empty list ({self.item_format}), got {value!r} for seq {key}"
            value0 = value[0]
            assert (
                isinstance(value0, tuple)
                and len(value0) == 2
                and isinstance(value0[0], float)
                and isinstance(value0[1], str)
            ), f"{self}: expected (score,text) tuples ({self.item_format}), got {value0!r} ({type(value0)})"
        # else:
        #     raise ValueError(f"invalid item_format {self.item_format!r}")

        for key, value in data.items():
            assert len(value) == 1  # just one score
            score, text = value[0]
            assert isinstance(score, float), f"{self}: expected float score, got {score!r} ({type(score)})"
            assert isinstance(text, str), f"{self}: expected str text, got {text!r} ({type(text)})"

        # Calculate average and median.
        scores = [value[0][0] for value in data.values()]
        avg_score = sum(scores) / len(scores)
        median_score = sorted(scores)[len(scores) // 2]

        self.out_avg.set(avg_score)
        self.out_median.set(median_score)
