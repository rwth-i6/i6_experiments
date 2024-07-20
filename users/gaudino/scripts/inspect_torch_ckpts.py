
import torch

import dictdiffer

model_ckpt_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.CMIQklG8vBT6/output/models/epoch.2000.pt"

ilm_ckpt_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.JOcvv4RtMmC9/output/models/epoch.020.pt"

model_ckpt = torch.load(model_ckpt_path, map_location="cpu")
ilm_ckpt = torch.load(ilm_ckpt_path, map_location="cpu")


print("Model ckpt parameters:")
for k, v in model_ckpt["model"].items():
    print(f"{k}: {v.shape if hasattr(v, 'shape') else v}")

print("\nILM ckpt parameters:")
for k, v in ilm_ckpt["model"].items():
    print(f"{k}: {v.shape if hasattr(v, 'shape') else v}")


# Checks
print("\n-------------- checks -----------------")

diff_keys = []

for k, v in ilm_ckpt["model"].items():
    if not k in model_ckpt["model"]:
        print(f"Key {k} not found in model ckpt")
        continue
    if model_ckpt["model"][k].shape != v.shape:
        print(f"Key {k} has different shape in model ckpt")
        continue
    if not torch.allclose(model_ckpt["model"][k], v):
        print(f"Key {k} has different values in model ckpt")
        diff_keys.append(k)
        continue

print("-------------- checks done -----------------")
print("Different keys:")
print(diff_keys)




