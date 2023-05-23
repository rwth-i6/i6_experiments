from i6_core import rasr, returnn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sisyphus import tk
from i6_experiments.users.berger.util import ToolPaths

from .returnn import get_native_lstm_op


@dataclass
class LMData(ABC):
    scale: float

    @abstractmethod
    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        ...


@dataclass
class ArpaLMData(LMData):
    filename: tk.Path

    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        config = rasr.RasrConfig()
        config.type = "ARPA"
        config.file = self.filename

        return config


@dataclass
class NNLMData(LMData, ABC):
    vocab_file: tk.Path
    model_file: returnn.Checkpoint
    returnn_config: returnn.ReturnnConfig
    unknown_word: str = "<UNK>"

    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        compiled_graph = returnn.CompileTFGraphJob(
            self.returnn_config,
            returnn_python_exe=tool_paths.returnn_python_exe,
            returnn_root=tool_paths.returnn_root,
            blas_lib=tool_paths.blas_lib,
        ).out_graph

        config = rasr.RasrConfig()
        config.scale = self.scale
        config.vocab_file = self.vocab_file
        config.transform_output_negate = True
        config.vocab_unknown_word = self.unknown_word

        config.loader.type = "meta"
        config.loader.meta_graph_file = compiled_graph
        config.loader.saved_model_file = self.model_file
        config.loader.required_libraries = get_native_lstm_op(tool_paths)

        config.input_map.info_0.param_name = "word"
        config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

        config.output_map.info_0.param_name = "softmax"
        config.output_map.info_0.tensor_name = "output/output_batch_major"

        return config


@dataclass
class RNNLMData(NNLMData):
    min_batch_size: int = 4
    max_batch_size: int = 64
    opt_batch_size: int = 64
    allow_reduced_history: bool = True

    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        config = super().get_config(tool_paths)

        config.type = "tfrnn"

        config.min_batch_size = self.min_batch_size
        config.max_batch_size = self.max_batch_size
        config.opt_batch_size = self.opt_batch_size
        config.allow_reduced_history = self.allow_reduced_history

        return config


@dataclass
class TransformerLMData(NNLMData):
    max_batch_size: int = 64

    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        config = super().get_config(tool_paths)

        config.type = "simple-transformer"

        config.max_batch_size = self.max_batch_size

        return config
