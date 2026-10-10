import ast
import json
from typing import Iterator

from sisyphus import Job, Task, tk


class LinearCombineNbestScoresJob(Job):
    """Per-hypothesis linear combination of two parallel n-best score files into one n-best TextDict."""

    def __init__(
        self,
        *,
        nbest_file: tk.Path,
        scores_file: tk.Path,
        nbest_coeff: float,
        scores_coeff: float,
    ) -> None:
        self.nbest_file = nbest_file
        self.scores_file = scores_file
        self.nbest_coeff = float(nbest_coeff)
        self.scores_coeff = float(scores_coeff)

        self.out_scores = self.output_path("combined_scores.py")
        self.out_first_rank_rate = self.output_var("first_rank_rate.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.nbest_file.get_path(), "rt") as f:
            nbest = ast.literal_eval(f.read())
        with open(self.scores_file.get_path(), "rt") as f:
            other = ast.literal_eval(f.read())
        assert set(nbest) == set(other), f"{len(set(nbest) ^ set(other))} seq tags differ between the two score files"

        first_rank = 0
        with open(self.out_scores.get_path(), "wt") as f:
            f.write("{\n")
            for tag, entries in nbest.items():
                other_entries = other[tag]
                assert len(entries) == len(other_entries), tag
                combined = []
                for (a, text), (b, other_text) in zip(entries, other_entries):
                    assert text == other_text, f"{tag}: hyp text mismatch"
                    combined.append(
                        (
                            self.nbest_coeff * float(a) + self.scores_coeff * float(b),
                            text,
                        )
                    )
                if combined:
                    best = max(range(len(combined)), key=lambda i: combined[i][0])
                    first_rank += int(best == 0)
                f.write(f"{str(tag)!r}: {combined!r},\n")
            f.write("}\n")
        self.out_first_rank_rate.set(first_rank / max(len(nbest), 1))


class ReadRaceScaleJob(Job):
    """The scale a first-pass race chose and the band of its survivors, as sisyphus variables."""

    def __init__(self, *, race_result: tk.Path) -> None:
        self.race_result = race_result
        self.out_scale = self.output_var("scale.txt")
        self.out_band = self.output_var("band.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.race_result.get_path(), "rt") as f:
            result = json.load(f)
        self.out_scale.set(float(result["chosen"]))
        self.out_band.set(tuple(float(scale) for scale in result["band"]))
