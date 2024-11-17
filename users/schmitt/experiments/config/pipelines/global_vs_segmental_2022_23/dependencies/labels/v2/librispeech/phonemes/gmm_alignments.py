from sisyphus import Path, tk
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.flow import FlowNetwork
from i6_core.rasr.feature_scorer import DiagonalMaximumScorer
from i6_core.lda.flow import add_context_flow
from i6_core.features.common import basic_cache_flow, add_linear_transform
from i6_core.features.extraction import FeatureExtractionJob
from i6_core.mm.alignment import AlignmentJob

from typing import Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.phonemes import LibrispeechPhonemes41, LibrispeechWords
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LIBRISPEECH_CORPUS
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import \
  SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import \
  LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import \
  SegmentalModelHyperparameters


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
    self.mfcc_mixture_file = Path(
      "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/mm/mixtures/EstimateMixturesJob.accumulate.lFTcG3mJGAaH/output/am.mix"
    )
    self.vtln_mixture_file = Path(
      ""
    )
    self.lda_matrix_path = Path("/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lda/estimate/EstimateLDAMatrixJob.bZslEyHe5P64/output/lda.matrix")
    self.corpus_paths = LIBRISPEECH_CORPUS.corpus_paths_wav
    self.segment_paths = LIBRISPEECH_CORPUS.segment_paths
    self.vtln_feature_caches = {
      "train": Path("/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"),
      "dev-clean": self._get_vtln_feature_cache_path("dev-clean")
    }
    self.mfcc_feature_caches = {
      "dev-clean": self._get_mfcc_feature_cache_path("dev-clean")
    }

    self.alignment_caches = {
      "train": Path("/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"),
    }

  def _get_vtln_feature_cache_path(self, corpus_key: str):
    return FeatureExtractionJob(
      self._get_feature_extraction_crp(corpus_key),
      LibrispeechGmmAlignment._get_vtln_feature_flow(),
      port_name_mapping={"features": "vtln"},
      job_name="VTLN"
    ).out_feature_path["vtln"]

  def _get_mfcc_feature_cache_path(self, corpus_key: str):
    return FeatureExtractionJob(
      self._get_feature_extraction_crp(corpus_key),
      LibrispeechGmmAlignment._get_mfcc_feature_flow(),
      port_name_mapping={"features": "mfcc"},
      job_name="MFCC"
    ).out_feature_path["mfcc"]

  def _get_alignment_cache_bundle(self, corpus_key: str):
    mfcc_feature_flow = basic_cache_flow(self.mfcc_feature_caches[corpus_key])
    mfcc_feature_flow = add_context_flow(mfcc_feature_flow)
    mfcc_feature_flow = add_linear_transform(mfcc_feature_flow, matrix_path=self.lda_matrix_path)
    return AlignmentJob(
      self._get_alignment_crp(corpus_key),
      feature_flow=mfcc_feature_flow,
      feature_scorer=self._get_alignment_feature_scorer(self.mfcc_mixture_file),
    ).out_alignment_bundle

  def _get_feature_extraction_crp(self, corpus_key: str):
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfig()
    corpus_config.file = self.corpus_paths[corpus_key]
    corpus_config.capitalize_transcriptions = False
    corpus_config.progress_indication = "global"
    corpus_config.warn_about_unexpected_elements = True
    if self.segment_paths[corpus_key] is not None:
      corpus_config.segments.file = self.segment_paths[corpus_key]
    crp.corpus_config = corpus_config

    return crp

  def _get_alignment_crp(self, corpus_key: str):
    crp = CommonRasrParameters()
    crp_add_default_output(crp, unbuffered=True)

    corpus_config = RasrConfig()
    corpus_config.file = self.corpus_paths[corpus_key]
    corpus_config.capitalize_transcriptions = False
    corpus_config.progress_indication = "global"
    corpus_config.warn_about_unexpected_elements = True
    if self.segment_paths[corpus_key] is not None:
      corpus_config.segments.file = self.segment_paths[corpus_key]
    crp.corpus_config = corpus_config

    lexicon_config = RasrConfig()
    lexicon_config.file = self.lexicon_file
    lexicon_config.normalize_pronunciation = False
    crp.lexicon_config = lexicon_config

    acoustic_model_config = RasrConfig()
    # allophones
    acoustic_model_config.allophones.add_all = False
    acoustic_model_config.allophones.add_from_file = self.allophone_file
    acoustic_model_config.allophones.add_from_lexicon = True
    # hmm
    acoustic_model_config.hmm.across_word_model = True
    acoustic_model_config.hmm.early_recombination = False
    acoustic_model_config.hmm.state_repetitions = 1
    acoustic_model_config.hmm.states_per_phone = 3
    # state-tying
    acoustic_model_config.state_tying.file = self.state_tying_file
    acoustic_model_config.state_tyting.type = "cart"
    # tdp
    acoustic_model_config.tdp.entry_m1.loop = "infinity"
    acoustic_model_config.tdp.entry_m2.loop = "infinity"
    acoustic_model_config.tdp.scale = 1.0
    acoustic_model_config.tdp["*"].exit = 0.0
    acoustic_model_config.tdp["*"].loop = 3.0
    acoustic_model_config.tdp["*"].forward = 0.0
    acoustic_model_config.tdp["*"].skip = "infinity"
    acoustic_model_config.tdp.silence.exit = 20.0
    acoustic_model_config.tdp.silence.loop = 0.0
    acoustic_model_config.tdp.silence.forward = 3.0
    acoustic_model_config.tdp.silence.skip = "infinity"
    crp.acoustic_model_config = acoustic_model_config

    return crp

  def _get_alignment_feature_scorer(self, mixtures: Path):
    return DiagonalMaximumScorer(scale=1.0, mixtures=mixtures)

  @staticmethod
  def _get_mfcc_normalization_flow(filterbank_warping_func: str):
    net = FlowNetwork()

    net.add_output("features")
    net.add_param("TASK")
    net.add_param("end-time")
    net.add_param("start-time")
    net.add_param("id")
    net.add_param("input-file")
    net.add_param("track")

    samples = net.add_node(
      "audio-input-file-wav",
      "samples",
      {"start-time": "$(start-time)", "end-time": "$(end-time)", "file": "$(input-file)"}
    )
    demultiplex = net.add_node("generic-vector-s16-demultiplex", "demultiplex", track="$(track)")
    net.link(samples, demultiplex)

    convert = net.add_node("generic-convert-vector-s16-to-vector-f32", "convert")
    net.link(demultiplex, convert)

    preemphesis = net.add_node("signal-preemphasis", "preemphasis", alpha=1.0)
    net.link(convert, preemphesis)

    window = net.add_node("signal-window", "window", length=0.025, shift=0.01, type="hamming")
    net.link(preemphesis, window)

    fft = net.add_node("signal-real-fast-fourier-transform", "fft", {"maximum-input-size": 0.025})
    net.link(window, fft)

    amplitude_spectrum = net.add_node(
      "signal-vector-alternating-complex-f32-amplitude", "amplitude-spectrum")
    net.link(fft, amplitude_spectrum)

    filterbank = net.add_node(
      "signal-filterbank",
      "filterbank",
      {"filter-width": 270.47838540079226, "warping-function": filterbank_warping_func}
    )
    net.link(amplitude_spectrum, filterbank)

    nonlinear = net.add_node("generic-vector-f32-log-plus", "nonlinear", value=1e-10)
    net.link(filterbank, nonlinear)

    cepstrum = net.add_node("signal-cosine-transform", "cepstrum", {"nr-outputs": 16})
    net.link(nonlinear, cepstrum)

    mfcc_normalization = net.add_node(
      "signal-normalization",
      "mfcc-normalization",
      length="infinity",
      right="infinity",
      type="mean-and-variance",
    )
    net.link(cepstrum, mfcc_normalization)

    return net, mfcc_normalization

  @staticmethod
  def _get_mfcc_feature_flow():
    net, mfcc_normalization = LibrispeechGmmAlignment._get_mfcc_normalization_flow("mel")
    net.link(mfcc_normalization, "network:features")
    return net

  @staticmethod
  def _get_vtln_feature_flow():
    net, mfcc_normalization = LibrispeechGmmAlignment._get_mfcc_normalization_flow(
      "nest(linear-2($input(alpha), 0.875), mel)")

    warping_factor = net.add_node(
      "generic-coprus-key-map",
      "warping-factor",
      {
        "key": "$(id)",
        "default-output" :1.0,
        "map-file": "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/vtln/train/ScoreFeaturesWithWarpingFactorsJob.vCPCdxOUFskH/output/warping_map.xml",
        "start-time": "$(start-time)"
      }
    )
    net.link(warping_factor, "filterbank:alpha")

    context_window = net.add_node(
      "signal-vector-f32-sequence-concatenation",
      "context-window",
      {
        "expand-timestamp": False,
        "margin-condition": "present-not-empty",
        "max-size": 9,
        "right": 4,
      }
    )
    net.link(mfcc_normalization, context_window)

    linear_transform = net.add_node(
      "signal-matrix-multiplication-f32",
      "linear-transform",
      file="/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lda/estimate/EstimateLDAMatrixJob.QU5zVyDzzZ9W/output/lda.matrix"
    )
    net.link(context_window, linear_transform)

    return net


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
