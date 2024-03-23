from typing import Dict, Any

from returnn_common.datasets_old_2022_10.interface import DatasetConfig

class CustomHdfDataset(DatasetConfig):

    def __init__(self, hdf_file_path):
        super(CustomHdfDataset, self).__init__()
        self.hdf_file_path = hdf_file_path

    def get_main_dataset(self) -> Dict[str, Any]:
        """
        More generic function, when this API is used for other purpose than training,
        e.g. recognition, generating alignment, collecting some statistics, etc,
        on one specific dataset.
        """
        return {
            "class": "HDFDataset",
            "files": [self.hdf_file_path],
        }
