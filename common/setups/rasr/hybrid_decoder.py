__all__ = ["HybridDecoder"]

from typing import Dict, Optional

from sisyphus import tk

from .base_decoder import BaseDecoder
from .util import RasrInitArgs, ReturnnRasrDataInput, NnRecogArgs


class HybridDecoder(BaseDecoder):
    def __init__(
        self, rasr_binary_path: tk.Path, rasr_arch: str = "linux-x86_64-standard"
    ):
        """
        Class to perform Hybrid Decoding with a RETURNN model

        :param rasr_binary_path: path to the rasr binary folder
        :param rasr_arch: RASR compile architecture suffix
        """
        super().__init__(rasr_binary_path=rasr_binary_path, rasr_arch=rasr_arch)

        self.rasr_init_args = None
        self.eval_corpora = []
        self.eval_input_data = None
        self.nn_recog_args = None

    def init_system(
        self,
        rasr_init_args: RasrInitArgs,
        eval_data: Dict[str, ReturnnRasrDataInput],
        nn_recog_args: NnRecogArgs,
    ):
        self.rasr_init_args = rasr_init_args
        self.eval_input_data = eval_data
        self.eval_corpora.extend(list(eval_data.keys()))
        self.nn_recog_args = nn_recog_args

    def recognition(self):
        pass
