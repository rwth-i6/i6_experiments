from typing import Optional

import torch

from i6_models.config import ModelConfiguration
from i6_core.returnn import PtCheckpoint
from sisyphus import Job, Task

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings


class MemristorModelConversionJob(Job):
    __sis_hash_exclude__ = {
        "regrid_embedding": True,
        "joint_correction_settings": None,
        "joint_correction_layer": 3,
        "regrid_joint_net_7": False,
    }

    def __init__(
        self,
        checkpoint: PtCheckpoint,
        model_config: ModelConfiguration,
        model_class: torch.nn.Module,
        replace_params: dict = None,
        regrid_embedding: bool = True,
        joint_correction_settings: Optional[CycleCorrectionSettings] = None,
        joint_correction_layer: int = 3,
        regrid_joint_net_7: bool = False,
    ):
        if joint_correction_layer not in (1, 2, 3):
            raise ValueError("joint_correction_layer must be 1, 2, or 3")
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.model_class = model_class
        self.replace_params = replace_params
        self.regrid_embedding = regrid_embedding
        self.joint_correction_settings = joint_correction_settings
        self.joint_correction_layer = joint_correction_layer
        self.regrid_joint_net_7 = regrid_joint_net_7
        self.out_config = self.model_config.with_replaced(**self.replace_params)
        self.out_checkpoint = PtCheckpoint(self.output_path("memristor_converted_model.pt"))
        self.rqmt = {"cpu": 48, "mem": 48, "time": 96}
        self.set_keep_value(0)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        checkpoint_data = torch.load(self.checkpoint.path, map_location="cpu")
        print(f"Converting model for memristor with config: {self.out_config}", flush=True)
        model = self.model_class(cfg=self.out_config)
        model.regrid_embedding = self.regrid_embedding
        # Conversion-only override: leave model-config serialization/hashes unchanged.
        model.joint_correction_settings = self.joint_correction_settings
        model.joint_correction_layer = self.joint_correction_layer
        model.load_state_dict(checkpoint_data["model"])
        model.eval()
        if self.regrid_joint_net_7:
            model.recalibrate_joint_net_7()
        model.prep_quant()
        torch.save({"model": model.state_dict()}, self.out_checkpoint.path)


def convert_model_for_memristor(
    checkpoint: PtCheckpoint,
    config: ModelConfiguration,
    model_class: torch.nn.Module,
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
    alias: Optional[str] = None,
    regrid_embedding: bool = True,
    joint_correction_settings: Optional[CycleCorrectionSettings] = None,
    joint_correction_layer: int = 3,
    regrid_joint_net_7: bool = False,
) -> tuple:
    base_converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    base_pos_enc_converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    # TODO: Always pass the base dac/adc settings hard coded right now. Otherwise, alternate settings result in hash changes.
    conversion_job = MemristorModelConversionJob(
        checkpoint=checkpoint,
        model_config=config,
        model_class=model_class,
        regrid_embedding=regrid_embedding,
        joint_correction_settings=joint_correction_settings,
        joint_correction_layer=joint_correction_layer,
        regrid_joint_net_7=regrid_joint_net_7,
        replace_params=dict(
            converter_hardware_settings=base_converter_hardware_settings,
            pos_enc_converter_hardware_settings=base_pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            num_cycles=num_cycles,
        ),
    )
    # change the config now
    replace_params = dict(
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        num_cycles=num_cycles,
    )
    conversion_job.out_config = conversion_job.out_config.with_replaced(**replace_params)
    if alias is not None:
        conversion_job.add_alias(f"memristor_conversion/{alias}")
    return conversion_job.out_checkpoint, conversion_job.out_config
