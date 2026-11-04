__all__ = ["SPEECH_LM_PYTHON_ENCODER_TYPE", "register_speech_lm_encoder_type"]

import numpy as np
import torch


SPEECH_LM_PYTHON_ENCODER_TYPE = "speech-lm-python"


def register_speech_lm_encoder_type(name: str = SPEECH_LM_PYTHON_ENCODER_TYPE) -> None:
    import librasr

    class SpeechLmPythonEncoder(librasr.Encoder):
        def __init__(self, config) -> None:
            super().__init__(config)

            checkpoint_path = config["checkpoint-path"]
            self.device = config["device"] or "cpu"

            import returnn.frontend as rf
            from returnn.util.basic import BehaviorVersion
            from speech_llm.prefix_lm.model.definitions.encoders.conformer import ConformerEncoderV1

            rf.select_backend_torch()
            BehaviorVersion.set_min_behavior_version(24)
            self.encoder = ConformerEncoderV1(
                enc_build_dict={
                    "class": "returnn.frontend.encoder.conformer.ConformerEncoder",
                    "input_layer": {
                        "class": "returnn.frontend.encoder.conformer.ConformerConvSubsample",
                        "out_dims": [32, 64, 64],
                        "filter_sizes": [(3, 3), (3, 3), (3, 3)],
                        "pool_sizes": [(1, 2)],
                        "strides": [(1, 1), (3, 1), (2, 1)],
                    },
                    "num_layers": 18,
                    "out_dim": 1024,
                    "encoder_layer": {
                        "class": "returnn.frontend.encoder.conformer.ConformerEncoderLayer",
                        "ff": {
                            "class": "returnn.frontend.encoder.conformer.ConformerPositionwiseFeedForward",
                            "activation": {"class": "rf.relu_square"},
                            "with_bias": False,
                        },
                        "num_heads": 8,
                    },
                },
                sampling_rate=16000,
                specaug_start=(5000, 15000, 25000),
                aux_loss_layers=(18,),
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            encoder_state_dict = {
                key[len("encoder.") :]: value for key, value in state_dict.items() if key.startswith("encoder.")
            }
            if not encoder_state_dict:
                raise RuntimeError(f"Could not find encoder weights in checkpoint {checkpoint_path!r}")

            missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            if missing_keys:
                raise RuntimeError(f"Missing keys while loading speech LLM encoder: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys while loading speech LLM encoder: {unexpected_keys}")

            self.encoder.eval()
            self.encoder.to(self.device)

        def encode(self, inputs: np.ndarray):
            if inputs.ndim != 2:
                raise ValueError(f"Expected encoder inputs with shape [T, F], got {inputs.shape}")
            if inputs.shape[1] != 1:
                raise ValueError(f"Expected raw audio with feature dimension 1, got shape {inputs.shape}")

            raw_audio = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0).squeeze(2)
            raw_audio_lens = torch.tensor([inputs.shape[0]], dtype=torch.int32, device=self.device)

            with torch.inference_mode():
                from .pytorch_modules import _encoder_forward

                encoder_output, _, _, _ = _encoder_forward(self, raw_audio, raw_audio_lens)
                encoder_output = encoder_output[0].detach().to(device="cpu", dtype=torch.float32)

            outputs = []
            num_outputs = int(encoder_output.shape[0])
            num_inputs = int(inputs.shape[0])
            if num_outputs == 0:
                return outputs
            for output_idx, encoding in enumerate(encoder_output.numpy()):
                input_start = output_idx * num_inputs // num_outputs
                input_end = (output_idx + 1) * num_inputs // num_outputs
                outputs.append((encoding, input_start, min(max(input_end, input_start + 1), num_inputs)))
            return outputs

    librasr.register_encoder_type(name, SpeechLmPythonEncoder)
