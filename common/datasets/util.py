from sisyphus import tk

from dataclasses import dataclass
from typing import Optional

from i6_core.meta.system import CorpusObject as _CorpusObject


@dataclass(frozen=True)
class CorpusObject(_CorpusObject):
    """
    Substitution of the old CorpusObject with a properly defined dataclass and strictly required members

    Contains additional corpus information needed for the RASR-Based pipelines

    :param corpus_file: bliss corpus file
    :param audio_format: file ending like definition, e.g. "wav", "flac" or "ogg"
    :param duration: duration of all segments in hours
    :param audio_dir: audio directory if paths in corpus bliss are relative (usually not needed)
    """

    corpus_file: tk.Path
    audio_format: str
    duration: float
    audio_dir: Optional[tk.Path] = None
