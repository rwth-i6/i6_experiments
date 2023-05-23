from dataclasses import dataclass
from typing import Dict, Optional

from i6_experiments.users.berger.util import ToolPaths
from i6_core import corpus, rasr, am, meta
from sisyphus import tk

from .rasr_lm_config import LMData


@dataclass
class LexiconConfig:
    filename: tk.Path
    normalize_pronunciation: bool
    add_all_allophones: bool
    add_allophones_from_lexicon: bool


@dataclass
class RasrDataInput:
    corpus_object: meta.CorpusObject
    lexicon: LexiconConfig
    lm: Optional[LMData] = None
    concurrent: int = 10
    stm: Optional[tk.Path] = None
    glm: Optional[tk.Path] = None


def get_crp_for_data_input(
    data: RasrDataInput,
    tool_paths: ToolPaths,
    am_args: Dict = {},
    base_crp: Optional[rasr.CommonRasrParameters] = None,
) -> rasr.CommonRasrParameters:
    crp = rasr.CommonRasrParameters(base_crp)

    rasr.crp_set_corpus(crp, data.corpus_object)
    crp.concurrent = data.concurrent
    crp.segment_path = corpus.SegmentCorpusJob(  # type: ignore
        data.corpus_object.corpus_file, data.concurrent
    ).out_segment_path

    if data.lm is not None:
        crp.language_model_config = data.lm.get_config(tool_paths)

    crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    crp.lexicon_config.file = data.lexicon.filename
    crp.lexicon_config.normalize_pronunciation = data.lexicon.normalize_pronunciation

    crp.acoustic_model_config = am.acoustic_model_config(**am_args)  # type: ignore
    crp.acoustic_model_config.allophones.add_all = data.lexicon.add_all_allophones  # type: ignore
    crp.acoustic_model_config.allophones.add_from_lexicon = data.lexicon.add_allophones_from_lexicon  # type: ignore

    return crp
