__all__ = ["ScoreJobType", "ScorableCorpus"]

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

from i6_core.corpus import CorpusToStmJob
from i6_core.recognition.scoring import Hub5ScoreJob, ScliteJob
from sisyphus import tk

from ...tools import sctk_binary_path


class ScoreJobType(Enum):
    Sclite = auto()
    Hub5 = auto()


@dataclass
class ScorableCorpus:
    corpus_name: str
    bliss_corpus_file: tk.Path
    stm_file: Optional[tk.Path] = None
    glm_file: Optional[tk.Path] = None
    score_job_type: ScoreJobType = ScoreJobType.Sclite

    def __post_init__(self) -> None:
        if self.stm_file is None:
            self.stm_file = CorpusToStmJob(self.bliss_corpus_file).out_stm_path

    def score_ctm(self, ctm_file: tk.Path, normalize_hf: bool = False) -> Union[ScliteJob, Hub5ScoreJob]:
        assert self.stm_file is not None
        ref_file = self.stm_file
        if normalize_hf:
            from .hf_scoring import NormalizeStmCtmJob

            normalized = NormalizeStmCtmJob(self.stm_file, ctm_file)
            ref_file, ctm_file = normalized.out_stm, normalized.out_ctm
        if self.score_job_type == ScoreJobType.Sclite:
            return ScliteJob(ref=ref_file, hyp=ctm_file, sort_files=True, sctk_binary_path=sctk_binary_path)
        elif self.score_job_type == ScoreJobType.Hub5:
            assert self.glm_file is not None
            return Hub5ScoreJob(ref=ref_file, glm=self.glm_file, hyp=ctm_file, sctk_binary_path=sctk_binary_path)
        else:
            raise ValueError(f"Invalid score job type {self.score_job_type}")
