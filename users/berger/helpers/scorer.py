from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Type
import i6_core.recognition as recognition
from i6_experiments.users.berger.recipe.converse.scoring import MeetEvalJob
from sisyphus import tk

ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob], Type[MeetEvalJob]]
ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]


@dataclass
class ScorerInfo:
    ref_file: Optional[tk.Path] = None
    job_type: ScoreJobType = recognition.ScliteJob
    score_kwargs: Dict = field(default_factory=dict)

    def get_score_job(self, ctm: tk.Path) -> ScoreJob:
        assert self.ref_file is not None
        return self.job_type(hyp=ctm, ref=self.ref_file, **self.score_kwargs)
