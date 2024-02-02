from dataclasses import dataclass
from typing import Dict, List
from i6_experiments.users.berger import helpers
from sisyphus import tk


@dataclass
class BasicSetupData:
    train_key: str
    dev_keys: List[str]
    test_keys: List[str]
    align_keys: List[str]
    train_data_config: Dict
    cv_data_config: Dict
    data_inputs: Dict[str, helpers.RasrDataInput]


@dataclass
class CTCSetupData(BasicSetupData):
    loss_corpus: tk.Path
    loss_lexicon: tk.Path


@dataclass
class PytorchCTCSetupData(BasicSetupData):
    pass


@dataclass
class HybridSetupData(BasicSetupData):
    pass


@dataclass
class SMSHybridSetupData(BasicSetupData):
    scoring_corpora: Dict[str, tk.Path]
    python_prolog: Dict
    num_classes: int


@dataclass
class BpeSetupData(BasicSetupData):
    bpe_lexicon: tk.Path
    forward_data_config: Dict[str, Dict]
