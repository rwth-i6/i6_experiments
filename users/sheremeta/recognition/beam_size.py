__all__ = ["GetBestBeamSizeByWerJob", "best_beam_size"]

import json
from typing import Dict

from sisyphus import Job, Task, tk


def best_beam_size(num_errors: Dict[int, int]) -> int:
    """
    :param num_errors: per beam size the word errors of its decode, all against one reference
    :return: the beam size with the fewest errors, the smallest one among equal counts
    """
    return min(sorted(num_errors), key=lambda beam_size: num_errors[beam_size])


class GetBestBeamSizeByWerJob(Job):
    """
    Picks the beam size whose decode has the fewest word errors against one reference, the smallest one among equal
    counts.
    """

    def __init__(self, num_errors: Dict[int, tk.Variable]):
        """
        :param num_errors: per beam size the word errors of its decode, for example the num_errors of a ScliteJob
        """
        self.num_errors = num_errors
        self.out_beam_size = self.output_var("beam_size")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        num_errors = {}
        for beam_size, errors in self.num_errors.items():
            count = errors.get()
            assert isinstance(count, int), f"beam size {beam_size}: {count!r} is not a count of word errors"
            num_errors[beam_size] = count
        beam_size = best_beam_size(num_errors)
        self.out_beam_size.set(beam_size)
        with open(self.out_summary.get_path(), "w") as f:
            summary = {"beam_size": beam_size, "num_errors": {str(b): e for b, e in sorted(num_errors.items())}}
            json.dump(summary, f, indent=2)
