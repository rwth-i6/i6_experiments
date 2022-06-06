from i6_experiments.users.schmitt.experiments.config.concat_seqs import concat_seqs, scoring
from sisyphus import *
import typing


def run_concat_seqs(ref_stm_paths: typing.Dict[str, Path], glm_path: Path, concat_nums: typing.List[int]):
  """Concat bundles of concat_nums seqs within a recording"""
  scoring.ScliteHubScoreJob.RefsStmFiles.update(ref_stm_paths)
  scoring.ScliteHubScoreJob.GlmFile = glm_path
  concat_jobs = {}
  for concat_num in concat_nums:
    concat_jobs[concat_num] = concat_seqs.ConcatSwitchboard.create_all_for_num(num=concat_num, register_output_prefix="concat_")

  return concat_jobs
