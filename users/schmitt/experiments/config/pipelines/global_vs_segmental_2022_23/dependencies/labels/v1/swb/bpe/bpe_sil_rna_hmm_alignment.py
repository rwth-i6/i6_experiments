from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe import BPE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.general import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.phonemes.phoneme_hmm_alignment import HMMPhoneme
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe_rna_alignment import RNABPE

from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob
from i6_experiments.users.schmitt.util.util import ModifySeqFileJob

from i6_experiments.users.schmitt.alignment.alignment import AugmentBPEAlignmentJob, AlignmentSplitSilenceJob, ReduceAlignmentJob, AlignmentStatisticsJob

from typing import Dict

from sisyphus import Path
from sisyphus.job_path import Variable


class RNABPESilBase(SegmentalLabelDefinition, BPE):
  """
  This alignment is a combination of our :class:`RNABPE` and :class:`HMMPhoneme` alignment, in which we try to augment
  our BPE alignment with silence from the phoneme alignment.
  """
  def __init__(
          self,
          ref_bpe: RNABPE,
          ref_phoneme: HMMPhoneme):
    super().__init__()

    self.ref_bpe = ref_bpe
    self.ref_phoneme = ref_phoneme

    self.augment_bpe_with_sil_jobs = {
      corpus_key: self._get_augment_bpe_with_sil_job(corpus_key) for corpus_key in ("train", "cv")
    }
    self.modify_seq_list_file_jobs = {
      corpus_key: self._get_modify_seq_list_file_job(corpus_key) for corpus_key in ("train", "devtrain", "cv")
    }

  @property
  def alias(self) -> str:
    return "rna-bpe-sil"

  def _get_augment_bpe_with_sil_job(self, corpus_key: str) -> AugmentBPEAlignmentJob:
    return AugmentBPEAlignmentJob(
      bpe_align_hdf=self.ref_bpe.alignment_paths[corpus_key],
      phoneme_align_hdf=self.ref_phoneme.alignment_paths[corpus_key],
      bpe_blank_idx=self.ref_bpe.model_hyperparameters.blank_idx,
      phoneme_blank_idx=self.ref_phoneme.model_hyperparameters.blank_idx,
      bpe_vocab=self.ref_bpe.vocab_path,
      phoneme_vocab=self.ref_phoneme.vocab_path,
      phoneme_lexicon=Path("/u/zhou/asr-exps/swb1/dependencies/train.lex.v1_0_3.ci.gz"),
      segment_file=self.ref_bpe.segment_paths[corpus_key],
      time_red_phon_align=1,
      time_red_bpe_align=6,
      time_rqtm= 2 if corpus_key == "cv" else 10,
      mem_rqmt=4 if corpus_key == "cv" else 6
    )

  def _get_modify_seq_list_file_job(self, corpus_key: str) -> ModifySeqFileJob:
    # since the devtrain seqs are a subset of the train seqs, we compare the devtrain seq file with the seqs
    # which are to be skipped in the train corpus
    seqs_to_skip_corpus_key = corpus_key if corpus_key != "devtrain" else "train"

    return ModifySeqFileJob(
      seq_file=self.ref_bpe.segment_paths[corpus_key],
      seqs_to_skip=self.augment_bpe_with_sil_jobs[seqs_to_skip_corpus_key].out_skipped_seqs_var)

  @property
  def train_segment_paths(self) -> Dict[str, Path]:
    return {
      "train": self.modify_seq_list_file_jobs["train"].out_seqs_file,
      "devtrain": self.modify_seq_list_file_jobs["devtrain"].out_seqs_file,
      "cv": self.modify_seq_list_file_jobs["cv"].out_seqs_file}

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    alignment_paths = {
      "train": self.augment_bpe_with_sil_jobs["train"].out_align,
      "cv": self.augment_bpe_with_sil_jobs["cv"].out_align}

    alignment_paths["devtrain"] = alignment_paths["train"]
    alignment_paths["cv300"] = alignment_paths["cv"]
    alignment_paths["cv_test"] = alignment_paths["cv"]

    return alignment_paths

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=1030, blank_idx=1031, target_num_labels=1032, sil_idx=0)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    json_to_rasr_job = BPEJSONVocabToRasrFormatsJob(
      self.vocab_path, blank_idx=self.model_hyperparameters.blank_idx)

    return RasrFormats(
      state_tying_path=json_to_rasr_job.out_state_tying,
      allophone_path=json_to_rasr_job.out_allophones,
      label_file_path=json_to_rasr_job.out_rasr_label_file,
      decoding_lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe-with-sil_lexicon"),
      realignment_lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/bpe_lexicon_phon"),
      blank_allophone_state_idx=4123)

  @property
  def vocab_path(self) -> Path:
    vocab_path = self.augment_bpe_with_sil_jobs["cv"].out_vocab
    return vocab_path

  @property
  def vocab_dict(self) -> Dict[str, Path]:
    return {
      "vocab_file": self.vocab_path,
      "bpe_file": self.bpe_codes_path}


class RNABPESil(SegmentalLabelDefinition, BPE):
  """
  This alignment is a combination of our :class:`RNABPE` and :class:`HMMPhoneme` alignment, in which we try to augment
  our BPE alignment with silence from the phoneme alignment.
  """
  def __init__(
          self,
          time_reduction: int,
          ref_bpe: RNABPE,
          ref_phoneme: HMMPhoneme,
          ref_rna_bpe_sil_base: RNABPESilBase):
    super().__init__()

    self.ref_bpe = ref_bpe
    self.ref_phoneme = ref_phoneme
    self.ref_rna_bpe_sil_base = ref_rna_bpe_sil_base

    self.time_reduction = time_reduction

    self.reduce_alignment_jobs = {
      corpus_key: self._get_reduce_alignment_job(corpus_key) for corpus_key in ("train", "cv")}

    self.modify_seq_list_file_jobs = {
      corpus_key: self._get_modify_seq_list_file_job(corpus_key) for corpus_key in ("train", "devtrain", "cv")}

  @property
  def alias(self) -> str:
    return "rna-bpe-sil"

  def _get_reduce_alignment_job(self, corpus_key: str) -> ReduceAlignmentJob:
    return ReduceAlignmentJob(
      hdf_align_path=self.ref_rna_bpe_sil_base.alignment_paths[corpus_key],
      segment_file=self.ref_rna_bpe_sil_base.segment_paths[corpus_key],
      blank_idx=self.model_hyperparameters.blank_idx,
      sil_idx=self.model_hyperparameters.sil_idx,
      reduction_factor=self.time_reduction)

  def _get_modify_seq_list_file_job(self, corpus_key: str) -> ModifySeqFileJob:
    # since the devtrain seqs are a subset of the train seqs, we compare the devtrain seq file with the seqs
    # which are to be skipped in the train corpus
    seqs_to_skip_corpus_key = corpus_key if corpus_key != "devtrain" else "train"

    return ModifySeqFileJob(
      seq_file=self.ref_rna_bpe_sil_base.segment_paths[corpus_key],
      seqs_to_skip=self._get_reduce_alignment_job(seqs_to_skip_corpus_key).out_skipped_seqs_var)

  @property
  def train_segment_paths(self) -> Dict[str, Path]:
    return {
      "train": self.modify_seq_list_file_jobs["train"].out_seqs_file,
      "devtrain": self.modify_seq_list_file_jobs["devtrain"].out_seqs_file,
      "cv": self.modify_seq_list_file_jobs["cv"].out_seqs_file}

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    alignment_paths = {
      "train": self.reduce_alignment_jobs["train"].out_align,
      "cv": self.reduce_alignment_jobs["cv"].out_align}

    alignment_paths["cv300"] = alignment_paths["cv"]
    alignment_paths["cv_test"] = alignment_paths["cv"]
    alignment_paths["devtrain"] = alignment_paths["train"]

    return alignment_paths

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return self.ref_rna_bpe_sil_base.model_hyperparameters

  @property
  def rasr_format_paths(self) -> RasrFormats:
    return self.ref_rna_bpe_sil_base.rasr_format_paths

  @property
  def vocab_path(self) -> Path:
    return self.ref_rna_bpe_sil_base.vocab_path

  @property
  def vocab_dict(self) -> Dict[str, Path]:
    return self.ref_rna_bpe_sil_base.vocab_dict


class RNABPESplitSil(SegmentalLabelDefinition, BPE):
  """
  This is the same alignment as :class:`RNABPESil` but the silence segments are split according to some maximum size.
  """
  def __init__(self, ref_bpe_sil: RNABPESil):
    super().__init__()

    self.ref_bpe_sil = ref_bpe_sil

    self.alignment_statistics_job = self._get_alignment_statistics_job(corpus_key="cv")
    self.alignment_split_sil_jobs = {
      corpus_key: self._get_alignment_split_sil_job(corpus_key) for corpus_key in("train", "cv")}

  @property
  def max_sil_len(self) -> Variable:
    return self.alignment_statistics_job.out_mean_non_sil_len_var

  def _get_alignment_split_sil_job(self, corpus_key: str):
    return AlignmentSplitSilenceJob(
      hdf_align_path=self.ref_bpe_sil.alignment_paths[corpus_key],
      segment_file=self.ref_bpe_sil.segment_paths[corpus_key],
      blank_idx=self.model_hyperparameters.blank_idx,
      sil_idx=self.model_hyperparameters.sil_idx,
      max_len=self.max_sil_len)

  def _get_alignment_statistics_job(self, corpus_key: str):
    return AlignmentStatisticsJob(
      alignment=self.ref_bpe_sil.alignment_paths[corpus_key],
      seq_list_filter_file=self.ref_bpe_sil.segment_paths[corpus_key],
      blank_idx=self.model_hyperparameters.blank_idx,
      silence_idx=self.model_hyperparameters.sil_idx)

  @property
  def train_segment_paths(self) -> Dict[str, Path]:
    return {
      "train": self.ref_bpe_sil.train_segment_paths["train"],
      "devtrain": self.ref_bpe_sil.train_segment_paths["devtrain"],
      "cv": self.ref_bpe_sil.train_segment_paths["cv"]}

  @property
  def alias(self) -> str:
    return "rna-bpe-split-sil"

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    alignment_paths = {
      "train": self.alignment_split_sil_jobs["train"].out_align,
      "cv": self.alignment_split_sil_jobs["cv"].out_align}

    alignment_paths["devtrain"] = alignment_paths["train"]
    alignment_paths["cv300"] = alignment_paths["cv"]
    alignment_paths["cv_test"] = alignment_paths["cv"]

    return alignment_paths

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return self.ref_bpe_sil.model_hyperparameters

  @property
  def rasr_format_paths(self) -> RasrFormats:
    return self.ref_bpe_sil.rasr_format_paths

  @property
  def vocab_path(self) -> Path:
    return self.ref_bpe_sil.vocab_path

  @property
  def vocab_dict(self) -> Dict[str, Path]:
    return self.ref_bpe_sil.vocab_dict
