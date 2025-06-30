from sisyphus import Path, tk

from typing import Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.phonemes import LibrispeechPhonemes41, LibrispeechWords
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import \
  SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import \
  LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import \
  SegmentalModelHyperparameters
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.alignment.alignment import GmmAlignmentToWordBoundariesJob

from ..tools_paths import RETURNN_EXE, RETURNN_ROOT


class LibrispeechGmmAlignment:
  """
    This is a GMM Alignment for Librispeech.
  """
  def __init__(self):
    self.allophone_file = Path(
      "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones")
    self.state_tying_file = Path(
      "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/cart/estimate/EstimateCartJob.aIIfPsPiYUiv/output/cart.tree.xml.gz")
    self.lexicon_file = Path(
      "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.SIIDsOAhK3bA/output/oov.lexicon.gz")
    self.corpus_file = Path(
      "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/corpus/transform/MergeCorporaJob.MFQmNDQlxmAB/output/merged.xml.gz")
    self.alignment_caches = {
      "train": Path("/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"),
    }


class LibrispeechGmmAlignmentConverted(LibrispeechPhonemes41, LibrispeechLabelDefinition, SegmentalLabelDefinition):
  def __init__(self):
    super().__init__()

    self._alignment_paths = None

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=-1, target_num_labels=41, sil_idx=1, blank_idx=0, target_num_labels_wo_blank=40)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  @property
  def bpe_codes_path(self) -> Path:
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
    assert "train" in value
    self._alignment_paths = value


class LibrispeechGmmWordAlignment(LibrispeechWords, LibrispeechLabelDefinition, SegmentalLabelDefinition):
  def __init__(self):
    super().__init__()

    self._alignment_paths = None

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=-1, target_num_labels=89116, sil_idx=1, blank_idx=0, target_num_labels_wo_blank=89115)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  @property
  def bpe_codes_path(self) -> Path:
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
    assert "train" in value
    self._alignment_paths = value


LIBRISPEECH_GMM_ALIGNMENT = LibrispeechGmmAlignment()
LIBRISPEECH_GMM_ALIGNMENT_CONVERTED = LibrispeechGmmAlignmentConverted()
LIBRISPEECH_GMM_WORD_ALIGNMENT = LibrispeechGmmWordAlignment()


def setup_gmm_alignment(ls_train_bliss_corpus):
  gmm_alignment = hdf.build_hdf_from_alignment(
    alignment_cache=LIBRISPEECH_GMM_ALIGNMENT.alignment_caches["train"],
    allophone_file=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
    state_tying_file=Path("/u/schmitt/experiments/segmental_models_2022_23_rf/state-tying"),
    returnn_python_exe=RETURNN_EXE,
    returnn_root=RETURNN_ROOT,
  )

  gmm_to_word_boundaries_job = GmmAlignmentToWordBoundariesJob(
    gmm_alignment_hdf=gmm_alignment,
    bliss_corpus=ls_train_bliss_corpus,
    allophone_path=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
  )
  gmm_to_word_boundaries_job.add_alias("datasets/gmm-word-boundaries/train")
  tk.register_output(gmm_to_word_boundaries_job.get_one_alias(), gmm_to_word_boundaries_job.out_hdf_align)

  LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths = {"train": gmm_to_word_boundaries_job.out_hdf_align}
  LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path = gmm_to_word_boundaries_job.out_vocab
