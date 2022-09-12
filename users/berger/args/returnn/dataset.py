from sisyphus import Path, tk
from typing import Optional


def get_lm_dataset_config(
    corpus_txt,
    vocab_file,
    num_classes_base,
    seq_ordering: str = "default",
    unknown_symbol: str = "[UNKNOWN]",
    partition_epoch: int = 1,
    orth_replace_map_file: Optional[Path] = None,
):
    return {
        "class": "LmDataset",
        "corpus_file": corpus_txt,
        "orth_symbols_map_file": vocab_file,
        "orth_replace_map_file": orth_replace_map_file or "",
        "word_based": True,
        "delayed_seq_data_start_symbol": "<s>",
        "seq_end_symbol": "<s>",
        "unknown_symbol": unknown_symbol,
        "auto_replace_unknown_symbol": True,
        "add_delayed_seq_data": True,
        "seq_ordering": seq_ordering,
        "partition_epoch": partition_epoch,
    }
