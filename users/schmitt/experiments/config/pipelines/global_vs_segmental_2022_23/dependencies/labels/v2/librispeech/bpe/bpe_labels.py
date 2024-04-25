from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.bpe.bpe import LibrispeechBPE10025
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import GlobalModelHyperparameters
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_ROOT, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.datasets.oggzip import get_dataset_dict as get_oggzip_dataset_dict
from i6_experiments.users.schmitt.alignment.silence import AddSilenceToBpeSeqsJob

from i6_core.returnn import ReturnnDumpHDFJob

from typing import Dict

from sisyphus import *


class LibrispeechBPE10025Labels(LibrispeechBPE10025, LibrispeechLabelDefinition, GlobalLabelDefinition):
  """
  These are the BPE labels of the SWB corpus.
  """
  def __init__(self):
    super().__init__()

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    return GlobalModelHyperparameters(
      sos_idx=0, target_num_labels=10025, sil_idx=None, target_num_labels_wo_blank=10025)


class LibrispeechBPE10025LabelsWithSilence(LibrispeechBPE10025, LibrispeechLabelDefinition, GlobalLabelDefinition):
  """
  These are the BPE labels of the SWB corpus.
  """
  def __init__(self, librispeech_bpe_10025_labels_instance: LibrispeechBPE10025Labels):
    super().__init__()

    self.librispeech_bpe_10025_labels_instance = librispeech_bpe_10025_labels_instance
    bpe_labels_hdf = {corpus_key: self.get_bpe_labels_as_hdf(corpus_key) for corpus_key in ("train",)}
    _add_silence_to_bpe_seqs_jobs = {
      corpus_key: AddSilenceToBpeSeqsJob(
        bpe_seqs_hdf=bpe_labels_hdf[corpus_key],
        bpe_vocab=self.librispeech_bpe_10025_labels_instance.vocab_path,
        phoneme_alignment_cache=LIBRISPEECH_GMM_ALIGNMENT.alignment_caches[corpus_key],
        allophone_file=LIBRISPEECH_GMM_ALIGNMENT.allophone_file,
        segment_file=self.segment_paths[corpus_key],
        returnn_python_exe=RETURNN_EXE_NEW,
        returnn_root=RETURNN_ROOT,
        silence_phone="[SILENCE]",
      ) for corpus_key in bpe_labels_hdf
    }
    self._label_paths = {
      corpus_key: _add_silence_to_bpe_seqs_jobs[corpus_key].out_hdf for corpus_key in bpe_labels_hdf
    }
    self._vocab_path = _add_silence_to_bpe_seqs_jobs["train"].out_vocab

    tk.register_output("bpe_labels_with_silence", self._label_paths["train"])

  def get_bpe_labels_as_hdf(self, corpus_key: str) -> Path:
    oggzip_dataset_dict = get_oggzip_dataset_dict(
      oggzip_path_list=self.oggzip_paths[corpus_key],
      bpe_file=self.librispeech_bpe_10025_labels_instance.bpe_codes_path,
      vocab_file=self.librispeech_bpe_10025_labels_instance.vocab_path,
      segment_file=self.segment_paths[corpus_key],
      fixed_random_subset=None,
      partition_epoch=1,
      pre_process=None,
      seq_ordering="default",
      epoch_wise_filter=None,
      seq_postfix=None,
    )
    oggzip_dataset_dict["data_map"] = {"data": ("zip_dataset", "classes")}
    hdf_file = ReturnnDumpHDFJob(
      oggzip_dataset_dict,
      returnn_python_exe=RETURNN_EXE_NEW,
      returnn_root=RETURNN_CURRENT_ROOT
    ).out_hdf
    return hdf_file

  @property
  def alias(self) -> str:
    return "bpe-with-silence"

  @property
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    return GlobalModelHyperparameters(
      sos_idx=0, target_num_labels=10026, sil_idx=10025, target_num_labels_wo_blank=10026)

  @property
  def label_paths(self) -> Dict[str, Path]:
    return self._label_paths
