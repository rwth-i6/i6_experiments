import numpy
from returnn.datasets.meta import AnythingDataset


class SomethingDataset(AnythingDataset):
    """
    Just like AnythingDataset, but with user-defined values
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self, seq_idx: int, key: str) -> numpy.ndarray:
        """data"""
        data = self._data_keys[key]
        if data.get("value") is None:
            return numpy.zeros([d or 1 for d in data["shape"]], dtype=data["dtype"])

        val = data["value"]
        # should be int or float
        assert isinstance(val, (int, float)), f"Value {val} is not int or float"

        return numpy.full(
            [d or 1 for d in data["shape"]],
            val,
            dtype=data["dtype"],
        )
