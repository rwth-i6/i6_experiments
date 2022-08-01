__all__ = ["HybridDecoder"]

from typing import Dict, Optional, Set, Tuple

from sisyphus import tk

from .base_decoder import BaseDecoder
from .util import RasrInitArgs, ReturnnRasrDataInput, NnRecogArgs


class HybridDecoder(BaseDecoder):
    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
    ):
        """
        Class to perform Hybrid Decoding with a RETURNN model

        :param rasr_binary_path: path to the rasr binary folder
        :param rasr_arch: RASR compile architecture suffix
        """
        super().__init__(rasr_binary_path=rasr_binary_path, rasr_arch=rasr_arch)

        self.returnn_root = returnn_root
        self.returnn_python_home = returnn_python_home
        self.returnn_python_exe = returnn_python_exe
        self.blas_lib = blas_lib

        self.rasr_init_args = None
        self.eval_corpora = []
        self.eval_input_data = None
        self.nn_recog_args = None

    def init_system(
        self,
        rasr_init_args: RasrInitArgs,
        eval_data: Dict[str, ReturnnRasrDataInput],
        nn_recog_args: NnRecogArgs,
        dev_test_mapping: Set[Tuple[str, str]],
    ):
        self.rasr_init_args = rasr_init_args
        self.eval_input_data = eval_data
        self.eval_corpora.extend(list(eval_data.keys()))
        self.nn_recog_args = nn_recog_args
        self.dev_test_mapping = dev_test_mapping

    def _compile_necessary_native_ops(self):
        pass

    def _compile_tf_graphs(self):
        pass

    def _get_base_flow(self):
        pass

    def _get_tf_flow(self):
        pass

    def _get_rasr_flow(self):
        pass

    def _get_lm_image_and_global_cache(self):
        pass

    def _get_prior_file(self):
        pass

    def _build_recog_loop(self):
        pass

    def recognition(self):
        # get necessary ops
        # get base flow
        # get tf
        # get flow for decoding
        # build dev-test map
        # construct crp
        pass
