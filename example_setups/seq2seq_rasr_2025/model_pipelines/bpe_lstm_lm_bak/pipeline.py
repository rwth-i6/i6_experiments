from dataclasses import dataclass

import black
from i6_core.returnn.forward import Optional
from i6_core.text.processing import WriteToTextFileJob
from i6_experiments.common.setups.returnn_pytorch.serialization import build_config_constructor_serializers_v2
from sisyphus import gs, tk

from .pytorch_modules import LstmLmConfig
from .subroutines.configs import TrainRoutineConfig
from .subroutines.train import train


@dataclass
class PipelineConfig:
    model_config: LstmLmConfig
    train_config: TrainRoutineConfig


def run_pipeline(config: PipelineConfig, name: Optional[str] = None, prefix: Optional[str] = None):
    out_dir = "/".join(filter(None, [prefix, "bpe_lstm_lm", name]))
    gs.ALIAS_AND_OUTPUT_SUBDIR = out_dir

    config_str, _ = build_config_constructor_serializers_v2(config)
    config_str = black.format_str(config_str.get(), mode=black.Mode(line_length=120))

    config_text_file = WriteToTextFileJob(content=config_str).out_file
    tk.register_output("pipeline_config.txt", config_text_file)

    train(config=config.train_config, model_config=config.model_config)
