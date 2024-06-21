__all__ = ["get_mono_transcription_priors"]

import numpy as np
from typing import Iterator, List
import pickle

from sisyphus import Job, Task

from i6_experiments.users.raissi.setups.common.decoder.config import PriorInfo, PriorConfig
from i6_experiments.users.raissi.setups.common.helpers.priors.util import write_prior_xml


pickles = {
    (
        1,
        False,
    ): "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/priors/daniel/monostate/monostate.pickle",
    (
        1,
        True,
    ): "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/priors/daniel/monostate/monostate.we.pickle",
    (
        3,
        False,
    ): "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/priors/daniel/threepartite/threepartite.pickle",
    (
        3,
        True,
    ): "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/priors/daniel/threepartite/threepartite.we.pickle",
}


class LoadTranscriptionPriorsJob(Job):
    def __init__(self, n: int, eow: bool):
        assert n in [1, 3]

        self.n = n
        self.eow = eow

        self.out_priors = self.output_path("priors.xml")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        file = pickles[(self.n, self.eow)]

        with open(file, "rb") as f:
            priors: List[float] = pickle.load(f)

        write_prior_xml(log_priors=np.log(priors), path=self.out_priors)


def get_mono_transcription_priors(states_per_phone: int, with_word_end: bool) -> PriorInfo:
    load_j = LoadTranscriptionPriorsJob(states_per_phone, with_word_end)
    return PriorInfo(center_state_prior=PriorConfig(file=load_j.out_priors, scale=0.0))