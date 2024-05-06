from dataclasses import dataclass

from sisyphus import tk

@dataclass
class SerializationAndHashArgs:
    package: str
    pytorch_model_import: str
    pytorch_train_step: str
