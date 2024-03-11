import os.path

import numpy as np
from sisyphus import Job, Task, tk
import sys
from typing import Callable, List

from i6_core.tools.git import CloneGitRepositoryJob


def get_confidence_intervals_repo():
    """
        Return the ConfidenceIntervals repo with static hash
    """
    confidence_interval_repo = CloneGitRepositoryJob(
        "https://github.com/luferrer/ConfidenceIntervals",
        commit="4e408cabfe6be86714b319fe506cc6af7bfbc2f4"

    ).out_repository.copy()
    confidence_interval_repo.hash_overwrite = "ConfidenceIntervalsRepo"
    return confidence_interval_repo

class BootstrapConfidenceJobTemplate(Job):

    def __init__(self):
        # load repo with hash overwrite to allow for future edits if necessary
        self.confidence_interval_repo = get_confidence_intervals_repo()

        self.out_mean = self.output_var("mean")
        self.out_min = self.output_var("min")
        self.out_max = self.output_var("max")
        self.out_max_interval_bound = self.output_var("max_interval_bound")

    def tasks(self):
        yield Task("run", mini_task=True)
        
    def compute_bootstrap(self, values: np.array, metric: Callable, conditions: List[int]):
        """

        """
        sys.path.insert(0, self.confidence_interval_repo.get_path())
        from confidence_intervals import evaluate_with_conf_int

        mean, (min, max) = evaluate_with_conf_int(values, metric, conditions=conditions)
        max_interval_boundary = np.max(np.abs([mean - min, max - mean]))

        print("mean: %f" % mean)
        print("min: %f" % min)
        print("max: %f" % max)
        print("max interval bound: %f" % max_interval_boundary)

        self.out_mean.set(mean)
        self.out_max.set(max)
        self.out_min.set(min)
        self.out_max_interval_bound.set(max_interval_boundary)


class ScliteBootstrapConfidenceJob(BootstrapConfidenceJobTemplate):

    def __init__(self, sclite_report_folder: tk.Path):
        super().__init__()
        self.sclite_report_folder = sclite_report_folder

    def run(self):

        def mean_metric(samples):
            return np.mean(samples)

        pra_file = os.path.join(self.sclite_report_folder.get(), "sclite.pra")
        wers = []
        speaker_ids = []
        speakers_dict = {}
        reading_speakers = False
        errors = []
        references = []
        for line in open(pra_file, "rt").readlines():
            line = line.strip()
            if reading_speakers:
                split = line.split(":")
                if len(split) != 2:
                    reading_speakers = False
                    print("Read speakers:")
                    print(speakers_dict)
                else:
                    print(line)
                    speakers_dict[split[1].strip()] = int(split[0])
                    print("add speaker %s as %s" % (split[1], split[0]))
            else:
                if line.startswith("Speakers:"):
                    reading_speakers = True
                if line.startswith("id: "):
                    speaker = line.split("(")[-1].split("-")[0]
                    speaker_id = speakers_dict[speaker]
                    speaker_ids.append(speaker_id)
                if line.startswith("Scores: (#C #S #D #I)"):
                    scores = line[len("Scores: (#C #S #D #I) "):].split(" ")
                    scores = [float(score) for score in scores]
                    assert len(scores) == 4
                    # do not count insertions for total words
                    error = np.sum(scores[1:])
                    words = np.sum(scores[:3])
                    wer = error / words
                    wers.append(wer)
                    errors.append(error)
                    references.append(words)

        print(np.sum(errors))
        print(np.sum(references))

        self.compute_bootstrap(np.asarray(wers), mean_metric, conditions=speaker_ids)
