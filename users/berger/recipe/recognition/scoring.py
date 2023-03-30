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
        best_name, best_job = min(
            self.scoring_jobs.items(), key=lambda item: item[1].out_num_errors
        )

        self.out_best_name.set(best_name)
        self.out_best_job.set(best_job)
