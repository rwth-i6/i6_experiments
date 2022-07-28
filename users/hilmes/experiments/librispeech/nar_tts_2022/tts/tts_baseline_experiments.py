"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from returnn_common.nn import min_returnn_behavior_version
from copy import deepcopy
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.corpus import CorpusReplaceOrthFromReferenceCorpus, MergeCorporaJob
from i6_core.corpus.segments import SegmentCorpusJob
from i6_experiments.common.setups.returnn_common.serialization import (
  Collection,
  ExternData,
  Import,
  Network,
  NonhashedCode,
)
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_private.users.hilmes.nar_tts.phoneme_data import (
  TTSTrainingDatasets,
  TTSEvalDataset,
)
from i6_private.users.hilmes.nar_tts.models.default_vocoder import get_default_vocoder
from i6_private.users.hilmes.tools.tts import VerifyCorpus, MultiJobCleanup
from i6_private.users.hilmes.util.asr_evaluation import asr_evaluation
from i6_private.users.hilmes.nar_tts.phoneme_data import get_tts_datastreams_phoneme
from i6_private.users.hilmes.nar_tts.phoneme_data import (
  get_tts_datastreams_from_rasr_alignment,
)