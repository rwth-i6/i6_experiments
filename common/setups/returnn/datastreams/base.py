from sisyphus import tk
from typing import *


class Datastream:
    """
    Defines a "Datastream" for a setup, e.g. related to an entry in the "extern_data" dictionary
    of the RETURNN config.

    This can be raw audio, audio-features, text labels, speaker labels etc...
    """

    def __init__(self, available_for_inference: bool):
        """

        :param available_for_inference: "default" value for "available_for_inference"
        """
        self.available_for_inference = available_for_inference

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None) -> Dict[str, Any]:
        """
        generates an option dictionary to be used as value of an `extern_data` entry.

        :param available_for_inference: allows to overwrite "available_for_inference" directly
        """
        opts = {
            "available_for_inference": available_for_inference
            if available_for_inference is not None
            else self.available_for_inference,
        }
        return opts


class FeatureDatastream(Datastream):
    """
    Defines a datastream for an arbitrary feature, e.g. from an HDFDataset
    """

    def __init__(
        self,
        available_for_inference: bool,
        feature_size: Union[tk.Variable, int],
    ):
        """
        :param available_for_inference:
        :param feature_size: feature dimension
        """
        super().__init__(available_for_inference)
        self.feature_size = feature_size

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """
        :param available_for_inference:
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(available_for_inference=available_for_inference),
            "shape": (None, self.feature_size),
            "dim": self.feature_size,
        }
        d.update(kwargs)
        return d
