from dataclasses import dataclass
from typing import Dict, List
from i6_experiments.common.setups.rasr.util.rasr import RasrDataInput
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
    data_inputs: Dict[str, RasrDataInput]
