from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from sisyphus import tk

from i6_core import rasr, returnn
from i6_experiments.users.berger.util import ToolPaths

from .returnn import get_native_lstm_op


@dataclass
class LMData(ABC):
    scale: float

    @abstractmethod
    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        ...

    def get_lookahead_config(self, tool_paths: ToolPaths) -> Optional[rasr.RasrConfig]:
        return None


@dataclass
class ArpaLMData(LMData):
    filename: tk.Path

    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        config = rasr.RasrConfig()
        config.type = "ARPA"
        config.file = self.filename
        config.scale = self.scale

        return config


@dataclass
class NNLMData(LMData, ABC):
    vocab_file: tk.Path
    model_file: returnn.Checkpoint
    returnn_config: Optional[returnn.ReturnnConfig] = None
    graph_file: Optional[tk.Path] = None
    unknown_word: str = "<UNK>"
    lookahead_lm: Optional[LMData] = None

    def _get_graph(self, tool_paths: ToolPaths) -> tk.Path:
        if self.graph_file is not None:
            return self.graph_file
        assert self.returnn_config is not None, "Must specify either a graph .meta file or a returnn config"
        return returnn.CompileTFGraphJob(
            self.returnn_config,
            returnn_python_exe=tool_paths.returnn_python_exe,
            returnn_root=tool_paths.returnn_root,
            blas_lib=tool_paths.blas_lib,
        ).out_graph

    def get_config(self, tool_paths: ToolPaths) -> rasr.RasrConfig:
        config = rasr.RasrConfig()
        config.scale = self.scale
        config.vocab_file = self.vocab_file
        config.transform_output_negate = True
        config.vocab_unknown_word = self.unknown_word

        config.loader.type = "meta"
        config.loader.meta_graph_file = self._get_graph(tool_paths=tool_paths)
        config.loader.saved_model_file = self.model_file
        config.loader.required_libraries = get_native_lstm_op(tool_paths)

        config.input_map.info_0.param_name = "word"
        config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

        config.output_map.info_0.param_name = "softmax"
        config.output_map.info_0.tensor_name = "output/output_batch_major"

        return config

    def get_lookahead_config(self, tool_paths: ToolPaths) -> Optional[rasr.RasrConfig]:
        if self.lookahead_lm is None:
            return None
        return self.lookahead_lm.get_config(tool_paths)


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
