from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob

from sisyphus import *


class RasrFormats:
  def __init__(
          self,
          state_tying_path: Path,
          allophone_path: Path,
          label_file_path: Path,
          bpe_no_phoneme_lexicon_path: Path,
          arpa4gram_bpe_phoneme_lexicon_path: Path,
          tfrnn_lm_bpe_phoneme_lexicon_path: Path,
          blank_allophone_state_idx: int,
          arpa_lm_image_path: Path = None,
  ):

    self.state_tying_path = state_tying_path
    self.allophone_path = allophone_path
    self.label_file_path = label_file_path
    self.bpe_no_phoneme_lexicon_path = bpe_no_phoneme_lexicon_path
    self.arpa4gram_bpe_phoneme_lexicon_path = arpa4gram_bpe_phoneme_lexicon_path
    self.tfrnn_lm_bpe_phoneme_lexicon_path = tfrnn_lm_bpe_phoneme_lexicon_path
    self.blank_allophone_state_idx = blank_allophone_state_idx
    self.arpa_lm_image_path = arpa_lm_image_path
