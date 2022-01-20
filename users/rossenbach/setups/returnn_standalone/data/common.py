from typing import *


class Datastream():
    """
    Defines a "Datastream" for a RETURNN setup, meaning a single entry in the "extern_data" dictionary.
    """

    def __init__(self, available_for_inference: bool):
        self.available_for_inference = available_for_inference

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None) -> Dict[str, Any]:
        opts = {
            "available_for_inference": available_for_inference
            if available_for_inference is not None
            else self.available_for_inference,
        }
        return opts

    def as_returnn_data_opts(self, available_for_inference: Optional[bool] = None) -> Dict[str, Any]:
        """
        deprecated
        """
        return self.as_returnn_extern_data_opts(available_for_inference)
