# #!/usr/bin/env python3 -u

# import os
# import torch
# import torch.nn.functional as F
# import fairseq
# import soundfile as sf

# chkpnt = "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/models/w2v2/wav2vec_vox_new.pt"
# single_audio = "/u/corpora/speech/LibriSpeech/LibriSpeech/test-clean/61/70970/61-70970-0007.wav"


# def read_audio(fname):
#     wav, sr = sf.read(fname)
#     assert sr == 16000, f"Expected 16kHz, got {sr}"
#     return wav


# def main():
#     # Load model
#     models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([chkpnt])
#     model = models[0]
#     model.eval().cuda()

#     print("\n✅ Model loaded.")

#     # Count transformer layers
#     encoder_layers = model.encoder.layers
#     num_layers = len(encoder_layers)
#     print(f"🔍 Number of transformer layers: {num_layers}\n")

#     # Read and prepare audio
#     x = read_audio(single_audio)
#     source = torch.from_numpy(x).float().cuda()
#     if task.cfg.normalize:
#         source = F.layer_norm(source, source.shape)
#     source = source.view(1, -1)

#     with torch.no_grad():
#         for layer in range(num_layers + 20):  # +1 because wav2vec also supports layer=0 (before encoder)
#             try:
#                 res = model(source=source, mask=False, features_only=True, layer=layer)
#                 feat = res["x"].squeeze(0).cpu()
#                 print(f"✅ Layer {layer:2d}: Feature shape = {feat.shape}")
#             except:
#                 print(f"Layer {layer} does not exist")
#     print("\n🎉 Done testing all layers.")


# if __name__ == "__main__":
#     main()

import torch
import torchaudio
import torchaudio.transforms as T
import fairseq

MODEL_PT_PATH = (
    "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/models/w2v2/wav2vec_vox_new.pt"
)
AUDIO_FILE_PATH = "/u/corpora/speech/LibriSpeech/LibriSpeech/test-clean/61/70970/61-70970-0007.wav"

models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([MODEL_PT_PATH])
model = models[0]
model.eval()

waveform, sample_rate = torchaudio.load(AUDIO_FILE_PATH)

if sample_rate != 16000:
    resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

if waveform.ndim > 1 and waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
waveform = waveform.to(device)

layer_outputs = {}


def get_layer_output_hook(layer_name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            layer_outputs[layer_name] = output[0].detach()
        else:
            layer_outputs[layer_name] = output.detach()

    return hook


features_layer_18 = None

if (
    hasattr(model, "encoder")
    and hasattr(model.encoder, "layers")
    and isinstance(model.encoder.layers, torch.nn.ModuleList)
):
    for TARGET_LAYER_IDX in range(27):
        try:
            if TARGET_LAYER_IDX < len(model.encoder.layers):
                target_module = model.encoder.layers[TARGET_LAYER_IDX]
                hook_handle = target_module.register_forward_hook(get_layer_output_hook(f"layer_{TARGET_LAYER_IDX}"))

                with torch.no_grad():
                    padding_mask = torch.zeros(waveform.shape[0], waveform.shape[1], dtype=torch.bool, device=device)

                    try:
                        model_input = {"source": waveform, "padding_mask": padding_mask, "mask": False}
                        if "features_only" in model.forward.__code__.co_varnames:
                            model_input["features_only"] = False
                        if "output_layer" in model.forward.__code__.co_varnames:  # For some specific ASR models
                            model_input["output_layer"] = None  # Ensure full forward pass

                        model(**model_input)

                    except Exception as e:
                        # Fallback for models that might not accept all above args or have different structure
                        # This might happen if `model` is not the base Wav2Vec2Model but a downstream one like Wav2VecCtc
                        # Try a simpler call if the model is just the encoder part or has a simpler forward
                        if hasattr(model, "w2v_encoder") and hasattr(
                            model.w2v_encoder, "w2v_model"
                        ):  # Common for ASR wrapper models
                            model.w2v_encoder.w2v_model(source=waveform, padding_mask=padding_mask, mask=False)
                        else:  # Generic attempt
                            model(waveform, padding_mask)

                hook_handle.remove()
                features_layer_18 = layer_outputs.get(f"layer_{TARGET_LAYER_IDX}")

                if features_layer_18 is not None:
                    print(f"Shape of features from layer {TARGET_LAYER_IDX + 1}: {features_layer_18.shape}")
                else:
                    print(f"Could not extract features from layer {TARGET_LAYER_IDX + 1}.")

            else:
                print(
                    f"Error: Target layer index {TARGET_LAYER_IDX} is out of bounds for model with {len(model.encoder.layers)} encoder layers."
                )
        except:
            print(f"Layer {TARGET_LAYER_IDX} does not exist or is not accessible.")
else:
    print("Error: Model does not have the expected 'encoder.layers' structure (e.g., model.encoder.layers).")
    print("Please inspect your Fairseq model structure to correctly identify the target module for the hook.")
