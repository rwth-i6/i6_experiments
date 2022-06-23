__all__ = ["SimpleTfNeuralLmRasrConfig"]

from typing import Optional

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn


class SimpleTfNeuralLmRasrConfig:
    def __init__(
        self,
        vocab_path: tk.Path,
        meta_graph_path: tk.Path,
        checkpoint: returnn.Checkpoint,
        *,
        scale: Optional[float] = None,
        unknown_symbol: str = "<UNK>",
        transform_output_negate: bool = True,
        output_layer_type: str = "softmax",
        libraries: Optional[tk.Path] = None,
        max_batch_size: int = 128,
    ):
        self.vocab_path = vocab_path
        self.meta_graph_path = meta_graph_path
        self.checkpoint = checkpoint
        self.scale = scale
        self.unknown_symbol = unknown_symbol
        self.transform_output_negate = transform_output_negate
        self.output_layer_type = output_layer_type
        self.libraries = libraries
        self.max_batch_size = max_batch_size

    def get(self) -> rasr.RasrConfig:
        lm_config = rasr.RasrConfig()

        lm_config.type = "simple-tf-neural"
        lm_config.vocab_file = self.vocab_path
        lm_config.vocab_unknown_word = self.unknown_symbol
        lm_config.transform_output_negate = self.transform_output_negate

        lm_config.loader.type = "meta"
        lm_config.loader.meta_graph_file = self.meta_graph_path
        lm_config.loader.saved_model_file = self.checkpoint
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

        lm_config.max_batch_size = self.max_batch_size

        return lm_config
