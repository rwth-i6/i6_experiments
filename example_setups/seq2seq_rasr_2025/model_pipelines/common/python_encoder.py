__all__ = [
    "get_pytorch_encoder_serializers",
    "get_rasr_python_encoder_init_hook_serializer",
    "register_pytorch_encoder_type",
]

from typing import List, Type

import numpy as np
import torch
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.returnn_pytorch.serialization import build_config_constructor_serializers_v2
from i6_experiments.common.setups.serialization import Collection, Import, SerializerObject
from i6_models.config import ModelConfiguration, ModuleType
from sisyphus.hash import sis_hash_helper


def register_pytorch_encoder_type(
    name: str,
    model_class: Type[torch.nn.Module],
    model_config: ModelConfiguration,
    checkpoint_path: str,
) -> None:
    import librasr

    class PyTorchEncoder(librasr.Encoder):
        def __init__(self, config) -> None:
            super().__init__(config)

            self.device = config["device"] or "cpu"
            self.model = model_class(cfg=model_config, epoch=1)

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                raise RuntimeError(f"Missing keys while loading {model_class.__name__}: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys while loading {model_class.__name__}: {unexpected_keys}")

            self.model.eval()
            self.model.to(self.device)

        def encode(self, inputs: np.ndarray):
            if inputs.ndim != 2:
                raise ValueError(f"Expected encoder inputs with shape [T, F], got {inputs.shape}")

            features = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
            features_size = torch.tensor([inputs.shape[0]], dtype=torch.int32, device=self.device)

            with torch.inference_mode():
                output = self.model(audio_samples=features, audio_samples_size=features_size)
                output_len = None
                if isinstance(output, tuple):
                    if len(output) > 1:
                        output_len = output[1]
                    output = output[0]

                output = output[0]
                if output_len is not None:
                    output = output[: int(output_len[0].item())]
                output = output.detach().to(device="cpu", dtype=torch.float32)

            outputs = []
            num_outputs = int(output.shape[0])
            num_inputs = int(inputs.shape[0])
            if num_outputs == 0:
                return outputs

            for output_idx, encoding in enumerate(output.numpy()):
                input_start = output_idx * num_inputs // num_outputs
                input_end = (output_idx + 1) * num_inputs // num_outputs
                outputs.append((encoding, input_start, min(max(input_end, input_start + 1), num_inputs)))
            return outputs

    librasr.register_encoder_type(name, PyTorchEncoder)


class _RasrPythonEncoderInitHook(SerializerObject):
    def __init__(self, registration_function_names: List[str], function_name: str = "register_rasr_python_encoders"):
        super().__init__()
        self.registration_function_names = registration_function_names
        self.function_name = function_name

    def get(self) -> str:
        if not self.registration_function_names:
            lines = [f"def {self.function_name}():\n"]
            lines.append("    pass\n")
        else:
            lines = [f"def {self.function_name}(\n"]
            for name in self.registration_function_names:
                lines.append(f"    {name}={name},\n")
            lines.append("):\n")
            for name in self.registration_function_names:
                lines.append(f"    {name}()\n")
        return "".join(lines)

    def _sis_hash(self) -> bytes:
        return sis_hash_helper(
            {
                "class": self.__class__.__name__,
                "registration_function_names": self.registration_function_names,
                "function_name": self.function_name,
            }
        )


class _PyTorchEncoderRegistration(SerializerObject):
    def __init__(
        self,
        *,
        encoder_type_name: str,
        function_name: str,
        model_class_name: str,
        model_config_variable_name: str,
        checkpoint_path: str,
    ):
        super().__init__()
        self.encoder_type_name = encoder_type_name
        self.function_name = function_name
        self.model_class_name = model_class_name
        self.model_config_variable_name = model_config_variable_name
        self.checkpoint_path = checkpoint_path

    def get(self) -> str:
        return (
            f"def {self.function_name}(\n"
            f"    register_pytorch_encoder_type=register_pytorch_encoder_type,\n"
            f"    {self.model_class_name}={self.model_class_name},\n"
            f"    {self.model_config_variable_name}={self.model_config_variable_name},\n"
            f"):\n"
            f"    register_pytorch_encoder_type(\n"
            f"        name={self.encoder_type_name!r},\n"
            f"        model_class={self.model_class_name},\n"
            f"        model_config={self.model_config_variable_name},\n"
            f"        checkpoint_path={self.checkpoint_path!r},\n"
            f"    )\n"
        )

    def _sis_hash(self) -> bytes:
        return sis_hash_helper(
            {
                "class": self.__class__.__name__,
                "encoder_type_name": self.encoder_type_name,
                "function_name": self.function_name,
                "model_class_name": self.model_class_name,
                "model_config_variable_name": self.model_config_variable_name,
                "checkpoint_path": self.checkpoint_path,
            }
        )


def get_pytorch_encoder_serializers(
    *,
    encoder_type_name: str,
    model_class: Type[ModuleType],
    model_config: ModelConfiguration,
    checkpoint: PtCheckpoint,
) -> Collection:
    cfg_variable_name = f"{encoder_type_name.replace('-', '_')}_cfg"
    register_function_name = f"register_{encoder_type_name.replace('-', '_')}"
    constructor_call, model_imports = build_config_constructor_serializers_v2(
        cfg=model_config,
        variable_name=cfg_variable_name,
    )

    return Collection(
        [
            Import(f"{model_class.__module__}.{model_class.__name__}"),
            *model_imports,
            constructor_call,
            Import(f"{register_pytorch_encoder_type.__module__}.{register_pytorch_encoder_type.__name__}"),
            _PyTorchEncoderRegistration(
                encoder_type_name=encoder_type_name,
                function_name=register_function_name,
                model_class_name=model_class.__name__,
                model_config_variable_name=cfg_variable_name,
                checkpoint_path=str(checkpoint),
            ),
        ]
    )


def get_rasr_python_encoder_init_hook_serializer(
    registration_function_names: List[str],
    function_name: str = "register_rasr_python_encoders",
) -> SerializerObject:
    return _RasrPythonEncoderInitHook(
        registration_function_names=registration_function_names,
        function_name=function_name,
    )
