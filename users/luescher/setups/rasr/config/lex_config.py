__all__ = ["LexiconRasrConfig"]

from dataclasses import dataclass

from sisyphus import tk

import i6_core.rasr as rasr


@dataclass()
class LexiconRasrConfig:
    lex_path: tk.Path
    normalize_pronunciation: bool

    def get(self) -> rasr.RasrConfig:
        lex_config = rasr.RasrConfig()

        lex_config.file = self.lex_path
        lex_config.normalize_pronunciation = self.normalize_pronunciation

        return lex_config
