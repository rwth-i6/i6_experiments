from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import i6_core.rasr as rasr
from i6_core import am, corpus, meta, rasr
from i6_experiments.users.berger.util import ToolPaths
from sisyphus import tk

from .rasr_lm_config import LMData


@dataclass
class LexiconConfig:
    filename: tk.Path
    normalize_pronunciation: bool
    add_all_allophones: bool
    add_allophones_from_lexicon: bool


@dataclass
class ScorableCorpusObject:
    corpus_file: Optional[tk.Path] = None  # bliss corpus xml
    audio_dir: Optional[tk.Path] = None  # audio directory if paths are relative (usually not needed)
    audio_format: Optional[str] = None  # format type of the audio files, see e.g. get_input_node_type()
    duration: Optional[float] = None  # duration of the corpus, is used to determine job time
    stm: Optional[tk.Path] = None
    glm: Optional[tk.Path] = None


@dataclass
class SeparatedCorpusObject:
    primary_corpus_file: tk.Path
    secondary_corpus_file: tk.Path
    mix_corpus_file: tk.Path
    audio_dir: Optional[tk.Path] = None
    audio_format: Optional[str] = None
    duration: Optional[float] = None
    stm: Optional[tk.Path] = None
    glm: Optional[tk.Path] = None

    @property
    def corpus_file(self) -> tk.Path:
        return self.primary_corpus_file

    def _get_corpus_object(self, corpus_file: tk.Path) -> ScorableCorpusObject:
        return ScorableCorpusObject(
            corpus_file=corpus_file,
            audio_dir=self.audio_dir,
            audio_format=self.audio_format,
            duration=self.duration,
            stm=self.stm,
            glm=self.glm,
        )

    def get_primary_corpus_object(self) -> ScorableCorpusObject:
        return self._get_corpus_object(self.primary_corpus_file)

    def get_secondary_corpus_object(self) -> ScorableCorpusObject:
        return self._get_corpus_object(self.secondary_corpus_file)

    def get_mix_corpus_object(self) -> ScorableCorpusObject:
        return self._get_corpus_object(self.mix_corpus_file)


@dataclass
class SeparatedCorpusHDFFiles:
    primary_features_files: List[tk.Path]
    secondary_features_files: List[tk.Path]
    mix_features_files: List[tk.Path]
    alignments_file: tk.Path
    segments: tk.Path


@dataclass
class RasrDataInput:
    corpus_object: Union[ScorableCorpusObject, SeparatedCorpusObject]
    lexicon: LexiconConfig
    lm: Optional[LMData] = None
    concurrent: int = 10


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
        crp.language_model_config = data.lm.get_config(tool_paths)  # type: ignore
        lookahead_config = data.lm.get_lookahead_config(tool_paths)
        if lookahead_config is not None:
            crp.lookahead_language_model_config = lookahead_config  # type: ignore

    crp.lexicon_config = rasr.RasrConfig()  # type: ignore
    crp.lexicon_config.file = data.lexicon.filename
    crp.lexicon_config.normalize_pronunciation = data.lexicon.normalize_pronunciation

    crp.acoustic_model_config = am.acoustic_model_config(**am_args)  # type: ignore
    crp.acoustic_model_config.allophones.add_all = data.lexicon.add_all_allophones  # type: ignore
    crp.acoustic_model_config.allophones.add_from_lexicon = data.lexicon.add_allophones_from_lexicon  # type: ignore

    return crp
