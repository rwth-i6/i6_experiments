from typing import Optional

import torch

from i6_models.config import ModelConfiguration
from i6_core.returnn import PtCheckpoint
from sisyphus import Job, Task

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

class MemristorModelConversionJob(Job):
    def __init__(self, checkpoint: PtCheckpoint, model_config: ModelConfiguration, model_class: torch.nn.Module, replace_params: dict = None):
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.model_class = model_class
        self.replace_params = replace_params
        self.out_config = self.model_config.with_replaced(**self.replace_params)
        self.out_checkpoint = PtCheckpoint(self.output_path("memristor_converted_model.pt"))
        self.rqmt = {"cpu": 2, "mem": 4, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        checkpoint_data = torch.load(self.checkpoint.path, map_location="cpu")
        print(f"Converting model for memristor with config: {self.out_config}", flush=True)
        model = self.model_class(cfg=self.out_config)
        model.load_state_dict(checkpoint_data["model"])
        model.prep_quant()
        torch.save({"model": model.state_dict()}, self.out_checkpoint.path)


def convert_model_for_memristor(checkpoint: PtCheckpoint, config: ModelConfiguration, model_class: torch.nn.Module, converter_hardware_settings: DacAdcHardwareSettings, pos_enc_converter_hardware_settings: DacAdcHardwareSettings, correction_settings: Optional[CycleCorrectionSettings], num_cycles: int) -> tuple:
    conversion_job = MemristorModelConversionJob(
        checkpoint=checkpoint,
        model_config=config,
        model_class=model_class,
        replace_params=dict(
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            num_cycles=num_cycles,
        ),
    )
    return conversion_job.out_checkpoint, conversion_job.out_config