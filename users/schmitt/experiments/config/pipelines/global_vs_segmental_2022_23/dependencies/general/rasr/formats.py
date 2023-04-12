from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob

from sisyphus import *


class RasrFormats:
  def __init__(
          self,
          state_tying_path: Path,
          allophone_path: Path,
          label_file_path: Path,
          decoding_lexicon_path: Path,
          realignment_lexicon_path: Path,
          blank_allophone_state_idx: int
  ):

    self.state_tying_path = state_tying_path
    self.allophone_path = allophone_path
    self.label_file_path = label_file_path
    self.decoding_lexicon_path = decoding_lexicon_path
    self.realignment_lexicon_path = realignment_lexicon_path
    self.blank_allophone_state_idx = blank_allophone_state_idx
