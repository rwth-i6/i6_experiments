from i6_private.users.schmitt.returnn.tools import BPEJSONVocabToRasrFormatsJob

from sisyphus import *


class RasrFormats:
  def __init__(
          self,
          vocab_path: Path,
          blank_idx: int,
          corpus_alias: str,
          label_alias: str,
          lexicon_path: Path,
          blank_allophone_state_idx: int
  ) -> None:

    json_to_rasr_job = BPEJSONVocabToRasrFormatsJob(vocab_path, blank_idx=blank_idx)
    job_alias = "rasr_formats/{corpus_name}/{label_name}".format(
      corpus_name=corpus_alias, label_name=label_alias)
    json_to_rasr_job.add_alias(job_alias)
    tk.register_output("{job_alias}/state_tying".format(job_alias=job_alias), json_to_rasr_job.out_state_tying)
    tk.register_output("{job_alias}/allophones".format(job_alias=job_alias), json_to_rasr_job.out_allophones)
    tk.register_output("{job_alias}/label_file".format(job_alias=job_alias), json_to_rasr_job.out_rasr_label_file)

    self.state_tying_path = json_to_rasr_job.out_state_tying
    self.allophone_path = json_to_rasr_job.out_allophones
    self.label_file_path = json_to_rasr_job.out_rasr_label_file

    self.lexicon_path = lexicon_path
    self.blank_allophone_state_idx = blank_allophone_state_idx
