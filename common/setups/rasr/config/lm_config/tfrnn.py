__all__ = ["TfRnnLmRasrConfig"]

from dataclasses import dataclass
from typing import List, Optional, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn


@dataclass()
class TfRnnLmRasrConfig:
    """
    Class for tf-rnn LM Params in RASR Config
    """

    vocab_path: tk.Path
    meta_graph_path: tk.Path
    returnn_checkpoint: returnn.Checkpoint
    scale: Optional[float] = None
    unknown_symbol: str = "<UNK>"
    transform_output_log: bool = True
    transform_output_negate: bool = True
    output_layer_type: str = "softmax"
    libraries: Optional[Union[tk.Path, List[tk.Path]]] = None
    state_manager: str = "transformer"
    softmax_adapter: Optional[str] = None
    common_prefix: bool = False

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()
        lm_config.type = "tfrnn"
        lm_config.vocab_file = self.vocab_path
        lm_config.vocab_unknown_word = self.unknown_symbol
        lm_config.transform_output_log = self.transform_output_log
        lm_config.transform_output_negate = self.transform_output_negate

        if self.scale is not None:
            lm_config.scale = self.scale

        lm_config.loader.type = "meta"
        lm_config.loader.meta_graph_file = self.meta_graph_path
        lm_config.loader.saved_model_file = self.returnn_checkpoint
        if self.libraries is not None:
            lm_config.loader.required_libraries = self.libraries

        lm_config.input_map.info_0.param_name = "word"
        lm_config.input_map.info_0.tensor_name = (
            "extern_data/placeholders/delayed/delayed"
        )
        lm_config.input_map.info_0.seq_length_tensor_name = (
            "extern_data/placeholders/delayed/delayed_dim0_size"
        )

        lm_config.output_map.info_0.param_name = self.output_layer_type
        lm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        lm_config.state_manager.type = self.state_manager

        if self.state_manager == "lstm":
            return lm_config
        elif self.state_manager == "transformer":
            lm_config.input_map.info_1.param_name = "state-lengths"
            lm_config.input_map.info_1.tensor_name = (
                "output/rec/dec_0_self_att_att/state_lengths"
            )

            if self.softmax_adapter is not None:
                lm_config.softmax_adapter.type = self.softmax_adapter
            if self.softmax_adapter in ("blas_nce", "quantized-blas-nce-16bit"):
                lm_config.output_map.info_0.tensor_name = "output/rec/decoder/add"
                lm_config.output_map.info_1.param_name = "weights"
                lm_config.output_map.info_1.tensor_name = "output/rec/output/W/read"
                lm_config.output_map.info_2.param_name = "bias"
                lm_config.output_map.info_2.tensor_name = "output/rec/output/b/read"
            if self.softmax_adapter == "quantized-blas-nce-16bit":
                lm_config.softmax_adapter.weights_bias_epsilon = 0.001

            if self.common_prefix:
                lm_config.state_manager.min_batch_size = 0
                lm_config.state_manager.min_common_prefix_length = 0

                for i in range(6):
                    c = lm_config.state_manager.var_map[f"item-{i}"]
                    c.var_name = f"output/rec/dec_{i}_self_att_att/keep_state_var:0"
                    c.common_prefix_initial_value = (
                        f"output/rec/dec_{i}_self_att_att/zeros_1:0"
                    )
                    c.common_prefix_initializer = (
                        f"output/rec/dec_{i}_self_att_att/common_prefix/Assign:0"
                    )
        else:
            raise NotImplementedError

        return lm_config
