__all__ = ["ScorableCorpus"]

from dataclasses import dataclass
from typing import Literal, Optional, Union

from i6_core.corpus import CorpusToStmJob
from i6_core.recognition.scoring import Hub5ScoreJob, ScliteJob
from sisyphus import tk

from ...tools import sctk_binary_path


@dataclass
class ScorableCorpus:
    corpus_name: str
    bliss_corpus_file: tk.Path
    stm_file: Optional[tk.Path] = None
    glm_file: Optional[tk.Path] = None
    score_job_type: Literal["Sclite", "Hub5"] = "Sclite"

    def __post_init__(self) -> None:
        if self.stm_file is None:
            self.stm_file = CorpusToStmJob(self.bliss_corpus_file).out_stm_path

    def score_ctm(self, ctm_file: tk.Path) -> Union[ScliteJob, Hub5ScoreJob]:
        assert self.stm_file is not None
        if self.score_job_type == "Sclite":
            return ScliteJob(ref=self.stm_file, hyp=ctm_file, sort_files=True, sctk_binary_path=sctk_binary_path)
        elif self.score_job_type == "Hub5":
            assert self.glm_file is not None
            return Hub5ScoreJob(ref=self.stm_file, glm=self.glm_file, hyp=ctm_file, sctk_binary_path=sctk_binary_path)
        else:
            raise ValueError(f"Invalid score job type {self.score_job_type}")
