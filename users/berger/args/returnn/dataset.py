from sisyphus import Path, tk
from typing import Any, Dict, List, Optional, Union


def get_lm_dataset_config(
    corpus_txt,
    vocab_file,
    seq_ordering: str = "default",
    unknown_symbol: str = "UNK",
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
        "auto_replace_unknown_symbol": False,
        "add_delayed_seq_data": True,
        "seq_ordering": seq_ordering,
        "partition_epoch": partition_epoch,
        "error_on_invalid_seq": False,
    }


class MetaDatasetBuilder:
    def __init__(self) -> None:
        self.datasets = {}
        self.data_map = {}
        self.control_dataset = ""

    def add_hdf_dataset(
        self,
        hdf_files: Union[tk.Path, List[tk.Path]],
        dataset_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not isinstance(hdf_files, list):
            hdf_files = [hdf_files]
        if dataset_config is None:
            dataset_config = {}
        dataset_config.update({"class": "HDFDataset", "use_cache_manager": True, "files": hdf_files})
        self.add_dataset(dataset_config=dataset_config, **kwargs)

    def add_dataset(
        self,
        name: str,
        dataset_config: Dict[str, Any],
        key_mapping: Dict[str, str],
        control: bool = False,
    ) -> None:
        self.datasets[name] = dataset_config
        for key, val in key_mapping.items():
            self.data_map[val] = (name, key)
        if control:
            self.control_dataset = name

    def get_dict(self) -> Dict[str, Any]:
        return {
            "class": "MetaDataset",
            "datasets": self.datasets,
            "data_map": self.data_map,
            "seq_order_control_dataset": self.control_dataset,
        }


def get_meta_dataset_from_hdf():
    pass
