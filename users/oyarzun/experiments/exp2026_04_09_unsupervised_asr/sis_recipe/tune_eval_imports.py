import ast

from i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.default_tools import (
    RETURNN_EXE,
    RETURNN_ONNX_EXE,
    RETURNN_ROOT,
    RETURNN_ONNX_ROOT,
)
from .pipeline import (
    search_single,
    get_checkpoint,
)
from typing import List, Optional, Dict, Any, List, Union, Iterator, Tuple, Callable
from dataclasses import dataclass, asdict
import copy
from enum import Enum

from sisyphus import tk, Task

from i6_core.returnn.compile import TorchOnnxExportJob
from i6_core.returnn.training import ReturnnTrainingJob, PtCheckpoint, ReturnnConfig
from i6_core.returnn.config import CodeWrapper
from i6_core.am.config import TdpValues, acoustic_model_config
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob, build_config_from_mapping
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.text.processing import PipelineJob
from i6_core.serialization.base import CallImport

from i6_experiments.common.setups.serialization import PartialImport
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.recognition.aed.beam_search import (
    DecoderConfig,
)
from i6_experiments.users.zeyer.datasets.score_results import (
    ScoreResultCollection,
    join_score_results,
    ScoreResult,
)
from i6_experiments.users.schmitt.lexicon.modification import (
    ReorderPhonemeInventoryByReturnnVocabJob,
    AddPhonemesAndLemmasToLexiconJob,
)
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import NonhashedCode
