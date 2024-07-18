from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import \
  RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import \
  SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.bpe.bpe import \
  LibrispeechBPE10025, LibrispeechBPE1056, LibrispeechBPE5048
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import \
  LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import \
  SegmentalModelHyperparameters
from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob
from i6_experiments.users.schmitt.rasr.lexicon import JsonSubwordVocabToLemmaLexiconJob, BlissCorpusToBpeLexiconJob
from i6_experiments.users.schmitt.alignment.alignment import AlignmentAddEosJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.lm_image import CreateLmImageJob

from typing import Dict
import copy
from abc import ABC

from sisyphus import *


class LibrispeechBpe10025CtcAlignment(LibrispeechBPE10025, LibrispeechLabelDefinition, SegmentalLabelDefinition):
  """
    This is a forced alignment from the auxiliary CTC model in Mohammad's global AED setup (5.6% WER).
  """
  def __init__(self):
    super().__init__()

    self._arpa4gram_bpe_phoneme_lexicon = None
    self._tfrnn_lm_bpe_phoneme_lexicon_path = None
    self._arpa_lm_image_path = None

    self._alignment_paths = None

  def _get_bpe_phoneme_lexicon(self, word_list_path: Path, unk_token: str) -> Path:
    subword_nmt_repo = CloneGitRepositoryJob(url="https://github.com/rsennrich/subword-nmt.git").out_repository
    return BlissCorpusToBpeLexiconJob(
      word_list_path=word_list_path,
      bpe_codes=self.bpe_codes_path,
      bpe_vocab=self.vocab_path,
      subword_nmt_repo=subword_nmt_repo,
      unk_token=unk_token,
    ).out_lexicon

  @property
  def arpa4gram_bpe_phoneme_lexicon_path(self) -> Path:
    if self._arpa4gram_bpe_phoneme_lexicon is None:
      self._arpa4gram_bpe_phoneme_lexicon = self._get_bpe_phoneme_lexicon(
        unk_token="<UNK>", word_list_path=self.lm_word_list_paths["arpa"])

    return self._arpa4gram_bpe_phoneme_lexicon

  @property
  def tfrnn_lm_bpe_phoneme_lexicon_path(self) -> Path:
    if self._tfrnn_lm_bpe_phoneme_lexicon_path is None:
      self._tfrnn_lm_bpe_phoneme_lexicon_path = self._get_bpe_phoneme_lexicon(
        unk_token="<unk>", word_list_path=self.lm_word_list_paths["kazuki-lstm"])

    return self._tfrnn_lm_bpe_phoneme_lexicon_path

  @property
  def arpa_lm_image_path(self) -> Path:
    if self._arpa_lm_image_path is None:
      self._arpa_lm_image_path = CreateLmImageJob(
        crp=RasrConfigBuilder.get_lm_image_crp(
          lexicon_path=self.tfrnn_lm_bpe_phoneme_lexicon_path,
          lm_file=self.arpa_lm_paths["arpa"],
          lm_type="ARPA",
        ),
        mem=8,
      ).out_image
    return self._arpa_lm_image_path

  @property
  def alias(self) -> str:
    return "global-att-aux-ctc-alignments"

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=10026, sil_idx=None, blank_idx=10025, target_num_labels_wo_blank=10025)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    json_to_rasr_job = BPEJSONVocabToRasrFormatsJob(
      Path(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"),
      blank_idx=10025)

    bpe_no_phoneme_lexicon_path = JsonSubwordVocabToLemmaLexiconJob(
      json_vocab_path=Path(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"),
      blank_idx=10025,
    ).out_lexicon

    return RasrFormats(
      state_tying_path=json_to_rasr_job.out_state_tying,
      allophone_path=json_to_rasr_job.out_allophones,
      label_file_path=json_to_rasr_job.out_rasr_label_file,
      bpe_no_phoneme_lexicon_path=bpe_no_phoneme_lexicon_path,
      arpa4gram_bpe_phoneme_lexicon_path=self.arpa4gram_bpe_phoneme_lexicon_path,
      tfrnn_lm_bpe_phoneme_lexicon_path=self.tfrnn_lm_bpe_phoneme_lexicon_path,
      blank_allophone_state_idx=40096,
      arpa_lm_image_path=self.arpa_lm_image_path,
    )

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    if self._alignment_paths is None:
      raise ValueError("Alignments first need to be set externally!")
    return self._alignment_paths

  @alignment_paths.setter
  def alignment_paths(self, value):
    assert isinstance(value, dict)
    assert self._alignment_paths is None, "Alignment paths are already set!"
    assert "train" in value and "cv" in value and "devtrain" in value
    self._alignment_paths = value


class LibrispeechBpe10025CtcAlignmentEos(LibrispeechBpe10025CtcAlignment):
  """
    This is a forced alignment from the auxiliary CTC model in Mohammad's global AED setup (5.6% WER).
  """
  def __init__(self, librispeech_bpe_10025_ctc_alignment_instance: LibrispeechBpe10025CtcAlignment):
    super().__init__()

    self.librispeech_bpe_10025_ctc_alignment_instance = librispeech_bpe_10025_ctc_alignment_instance
    self._alignment_paths = None

  @property
  def alias(self) -> str:
    return "global-att-aux-ctc-alignments-eos"

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    if self._alignment_paths is None:
      self._alignment_paths = {
        AlignmentAddEosJob(
          hdf_align_path=alignment_path,
          segment_file=self.segment_paths.get(corpus_key, None),
          blank_idx=self.model_hyperparameters.blank_idx,
          eos_idx=self.model_hyperparameters.sos_idx,
          returnn_python_exe=RETURNN_EXE,
          returnn_root=RETURNN_ROOT,
        ) for corpus_key, alignment_path in self.librispeech_bpe_10025_ctc_alignment_instance.alignment_paths.items()
      }
    return self._alignment_paths


class LibrispeechBpe1056Alignment(LibrispeechBPE1056, LibrispeechLabelDefinition, SegmentalLabelDefinition):
  """
    This is a forced alignment from the auxiliary CTC model in Mohammad's global AED setup (5.6% WER).
  """
  def __init__(self):
    super().__init__()

    self._alignment_paths = None

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=1057, sil_idx=None, blank_idx=0, target_num_labels_wo_blank=1056)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  @property
  def alias(self) -> str:
    return "att-transducer-alignment"

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    if self._alignment_paths is None:
      raise ValueError("Alignments first need to be set externally!")
    return self._alignment_paths

  @alignment_paths.setter
  def alignment_paths(self, value):
    assert isinstance(value, dict)
    assert self._alignment_paths is None, "Alignment paths are already set!"
    assert "train" in value and "cv" in value and "devtrain" in value
    self._alignment_paths = value


class LibrispeechBpe5048Alignment(LibrispeechBPE5048, LibrispeechLabelDefinition, SegmentalLabelDefinition, ABC):
  def __init__(self):
    super().__init__()

    self._alignment_paths = None

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  @property
  def alias(self) -> str:
    return "att-transducer-alignment"

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    if self._alignment_paths is None:
      raise ValueError("Alignments first need to be set externally!")
    return self._alignment_paths

  @alignment_paths.setter
  def alignment_paths(self, value):
    assert isinstance(value, dict)
    assert self._alignment_paths is None, "Alignment paths are already set!"
    assert "train" in value and "cv" in value and "devtrain" in value
    self._alignment_paths = value


class LibrispeechBpe5048AlignmentJointModel(LibrispeechBpe5048Alignment):
  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=5049, sil_idx=None, blank_idx=0, target_num_labels_wo_blank=5048)


class LibrispeechBpe5048AlignmentSepModel(LibrispeechBpe5048Alignment):
  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=5049, sil_idx=None, blank_idx=5048, target_num_labels_wo_blank=5048)
