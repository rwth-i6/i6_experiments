from typing import Dict, Generator
from sisyphus import Job, Task


class BestScoreJob(Job):
    """
    Provided a dict of names and scoring-jobs (e.g. Sclite), find the job with the lowest
    amount of errors and return the corresponding name and job.
    """

    def __init__(self, scoring_jobs: Dict[str, Job]) -> None:
        """
        :param scoring_jobs: Dict of scoring-jobs from which the best one is selected
        """
        super().__init__()

        self.scoring_jobs = scoring_jobs

        self.out_best_name = self.output_var("best_score_name")
        self.out_best_job = self.output_var("best_score_job")

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        best_name, best_job = min(self.scoring_jobs.items(), key=lambda item: item[1].out_num_errors)

        self.out_best_name.set(best_name)
        self.out_best_job.set(best_job)


class UpsampleCtmFileJob(Job):
    def __init__(self, in_ctm_file):
        self.in_ctm_file = in_ctm_file
        self.out_ctm_file = self.output_path("upsampledLattice.ctm")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        control_str = "<name> <track> <start> <duration> <word> <confidence>"
        with open(self.in_ctm_file.get_path(), "rt") as in_file, open(self.out_ctm_file.get_path(), "wt") as out_file:
            for in_line in in_file:
                prediction = in_line.split()
                if in_line.startswith(";;"):
                    out_line = in_line
                    curr_start, curr_duration = 0.0, 0.0
                else:
                    assert len(prediction) == 6, f"expected {control_str}, got {prediction}"
                    if curr_start == 0.0:
                        curr_start = float(prediction[2])
                    else:
                        curr_start += float(curr_duration)

                    curr_duration = float(prediction[3]) * 4
                    prediction[2] = f"{curr_start:.03f}"
                    prediction[3] = f"{curr_duration:.03f}"
                    out_line = " ".join(prediction)
                    out_line += "\n"

                out_file.write(out_line)
