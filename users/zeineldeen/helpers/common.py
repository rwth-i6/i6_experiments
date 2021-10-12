from typing import *


class Datastream():
    """
    Defines a "Datastream" for a RETURNN setup, meaning a single entry in the "extern_data" dictionary.
    """

    def __init__(self, available_for_inference: bool):
        self.available_for_inference = available_for_inference

    def as_returnn_data_opts(self) -> Dict[str, Any]:
        raise NotImplementedError
