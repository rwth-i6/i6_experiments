__all__ = ["LmData"]

from typing import Optional, Union

from sisyphus import Path


class LmData:
    def __init__(
        self,
        data_file: Path,
        vocabulary_file: Path,
        word_based: bool,
        seq_order: str,
        epoch_split: Union[int, float],
        unknown_symbol: str = "<UNK>",
        map_orthography_file: Union[str, Path] = "",
        start_end_symbol: str = "<sb>",
        data_type: Optional[str] = None,
    ):
        self.data_file = data_file
        self.vocabulary_file = vocabulary_file
        self.word_based = word_based
        self.seq_order = seq_order
        self.epoch_split = epoch_split
        self.unknown_symbol = unknown_symbol
        self.map_orthography_file = map_orthography_file
        self.start_end_symbol = start_end_symbol
        self.data_type = data_type

    def get_data_type(self):
        return self.data_type

    def get_dataset(self):
        return {
            "class": "LmDataset",
            "corpus_file": self.data_file,
            "orth_symbols_map_file": self.vocabulary_file,
            "orth_replace_map_file": self.map_orthography_file,
            "word_based": True,
            "seq_end_symbol": self.start_end_symbol,
            "auto_replace_unknown_symbol": False,
            "unknown_symbol": self.unknown_symbol,
            "add_delayed_seq_data": True,
            "delayed_seq_data_start_symbol": self.start_end_symbol,
            "seq_ordering": self.seq_order,
            "partition_epoch": self.epoch_split,
        }
