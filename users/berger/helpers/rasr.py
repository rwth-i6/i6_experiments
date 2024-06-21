from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import i6_core.rasr as rasr
from i6_core import am, corpus, meta, rasr
from i6_experiments.users.berger.util import ToolPaths
from i6_experiments.common.datasets.util import CorpusObject
from sisyphus import tk

from i6_experiments.users.berger.helpers.rasr_lm_config import ArpaLMData, LMData
from i6_experiments.users.berger.helpers.scorer import ScorerInfo


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


def convert_legacy_corpus_object_to_scorable(
    corpus_object: Union[meta.CorpusObject, CorpusObject],
) -> ScorableCorpusObject:
    return ScorableCorpusObject(
        corpus_file=corpus_object.corpus_file,
        audio_dir=corpus_object.audio_dir,
        audio_format=corpus_object.audio_format,
        duration=corpus_object.duration,
    )


def convert_legacy_corpus_object_dict_to_scorable(
    corpus_object_dict: Dict[str, Union[meta.CorpusObject, CorpusObject]],
) -> Dict[str, ScorableCorpusObject]:
    return {
        key: convert_legacy_corpus_object_to_scorable(corpus_object)
        for key, corpus_object in corpus_object_dict.items()
    }


@dataclass
class RasrDataInput:
    corpus_object: Union[ScorableCorpusObject, SeparatedCorpusObject]
    lexicon: LexiconConfig
    lm: Optional[LMData] = None
    concurrent: int = 10
    scorer: Optional[ScorerInfo] = None

    def create_lm_images(self, rasr_binary_path: tk.Path) -> None:
        if self.lm is None:
            return

        if isinstance(self.lm, ArpaLMData):
            self.lm.create_image(rasr_binary_path=rasr_binary_path, lexicon_file=self.lexicon.filename)

        if self.lm.lookahead_lm is not None and isinstance(self.lm.lookahead_lm, ArpaLMData):
            self.lm.lookahead_lm.create_image(rasr_binary_path=rasr_binary_path, lexicon_file=self.lexicon.filename)

    def get_crp(
        self,
        rasr_python_exe: tk.Path,
        rasr_binary_path: tk.Path,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        blas_lib: Optional[tk.Path] = None,
        am_args: Optional[Dict] = None,
    ) -> rasr.CommonRasrParameters:
        if am_args is None:
            am_args = {}

        crp = rasr.CommonRasrParameters()
        crp.python_program_name = rasr_python_exe  # type: ignore
        rasr.crp_add_default_output(crp)
        crp.set_executables(rasr_binary_path=rasr_binary_path)

        rasr.crp_set_corpus(crp, self.corpus_object)
        crp.concurrent = self.concurrent
        crp.segment_path = corpus.SegmentCorpusJob(  # type: ignore
            self.corpus_object.corpus_file, self.concurrent
        ).out_segment_path

        crp.lexicon_config = rasr.RasrConfig()  # type: ignore
        crp.lexicon_config.file = self.lexicon.filename  # type: ignore
        crp.lexicon_config.normalize_pronunciation = self.lexicon.normalize_pronunciation  # type: ignore

        crp.acoustic_model_config = am.acoustic_model_config(**am_args)  # type: ignore
        crp.acoustic_model_config.allophones.add_all = self.lexicon.add_all_allophones  # type: ignore
        crp.acoustic_model_config.allophones.add_from_lexicon = self.lexicon.add_allophones_from_lexicon  # type: ignore

        if self.lm is not None:
            lm_config = self.lm.get_config(
                returnn_python_exe=returnn_python_exe, returnn_root=returnn_root, blas_lib=blas_lib
            )  # type: ignore
            lookahead_lm_config = self.lm.get_lookahead_config(
                returnn_python_exe=returnn_python_exe, returnn_root=returnn_root, blas_lib=blas_lib
            )

            crp.language_model_config = lm_config  # type: ignore
            if lookahead_lm_config is not None:
                crp.lookahead_language_model_config = lookahead_lm_config  # type: ignore

        return crp
