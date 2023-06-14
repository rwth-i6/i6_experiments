from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.general import SegmentalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.phonemes.phonemes import Phonemes
from i6_experiments.users.schmitt.alignment.alignment import DumpPhonemeAlignJob
from i6_experiments.users.schmitt.rasr.convert import PhonJSONVocabToRasrFormatsJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from typing import Dict

from sisyphus import *


class HMMPhoneme(SegmentalLabelDefinition, Phonemes):
  """
  This is an HMM phoneme alignment by Zoltan Tüske. It is modified so that we use 84 phonemes
  (42 phonemes + EOW-augmentation) + NOISE/VOC-NOISE/LAUGHTER + Silence + Blank + SOS = 90 targets.
  """
  def _get_dump_phoneme_align_job(self, corpus_key):
    phon_align_extraction_config = RasrConfigBuilder.get_phon_align_extraction_config(
      corpus_path=self.corpus_paths[corpus_key],
      segment_path=self.segment_paths[corpus_key],
      feature_cache_path=self.feature_cache_paths[corpus_key],
      allophone_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/allophones"),
      lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/zhou-phoneme-transducer/lexicon/train.lex.v1_0_3.ci.gz"),
      alignment_cache_path=Path("/work/asr4/zhou/asr-exps/swb1/dependencies/zoltan-alignment/train.alignment.cache.bundle", cached=True)
    )
    dump_phon_align_job = DumpPhonemeAlignJob(
      rasr_config=phon_align_extraction_config,
      time_red=1,
      time_rqtm=10,
      rasr_exe=RasrExecutables.nn_trainer_path,
      mem_rqmt=8,
      state_tying_file=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states"))

    tk.register_output("temp-phon-align", dump_phon_align_job.out_align)

    return dump_phon_align_job

  @property
  def train_segment_paths(self) -> Dict[str, Path]:
    return LabelDefinition.get_default_train_segment_paths()

  @property
  def alias(self) -> str:
    return "hmm-phoneme"

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    alignment_paths = {
      "train": self._get_dump_phoneme_align_job(corpus_key="train").out_align,
      "devtrain": self._get_dump_phoneme_align_job(corpus_key="train").out_align,
      "cv": self._get_dump_phoneme_align_job(corpus_key="cv").out_align}

    alignment_paths["cv300"] = alignment_paths["cv"]
    alignment_paths["cv_test"] = alignment_paths["cv"]

    return alignment_paths

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=88, blank_idx=89, target_num_labels=90, sil_idx=0)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    json_to_rasr_job = PhonJSONVocabToRasrFormatsJob(
      self.vocab_path,
      blank_idx=self.model_hyperparameters.blank_idx)

    return RasrFormats(
      state_tying_path=json_to_rasr_job.out_state_tying,
      allophone_path=json_to_rasr_job.out_allophones,
      label_file_path=json_to_rasr_job.out_rasr_label_file,
      decoding_lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon"),
      realignment_lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/my_new_lex"),
      blank_allophone_state_idx=4119
    )

  @property
  def vocab_path(self) -> Path:
    return self._get_dump_phoneme_align_job(corpus_key="cv").out_phoneme_vocab

  @property
  def vocab_dict(self) -> Dict[str, Path]:
    return {
      "vocab_file": self.vocab_path}
