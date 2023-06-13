from dataclasses import dataclass
from typing import Dict, List
from i6_experiments.users.berger import helpers
from sisyphus import tk


@dataclass
class CTCSetupData:
    train_key: str
    dev_keys: List[str]
    test_keys: List[str]
    align_keys: List[str]
    train_data_config: Dict
    cv_data_config: Dict
    loss_corpus: tk.Path
    loss_lexicon: tk.Path
    data_inputs: Dict[str, helpers.RasrDataInput]


@dataclass
class PytorchCTCSetupData:
    train_key: str
    dev_keys: List[str]
    test_keys: List[str]
    align_keys: List[str]
    train_data_config: Dict
    cv_data_config: Dict
    data_inputs: Dict[str, helpers.RasrDataInput]


@dataclass
class SMSHybridSetupData:
    train_key: str
    dev_keys: List[str]
    test_keys: List[str]
    align_keys: List[str]
    train_data_config: Dict
    cv_data_config: Dict
    data_inputs: Dict[str, helpers.RasrDataInput]
    scoring_corpora: Dict[str, tk.Path]
    python_prolog: Dict
    num_classes: int
