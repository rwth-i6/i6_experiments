
from sisyphus import tk
from ..datasets import librispeech

for name, path in librispeech.lirispeech_ogg_zip_dict.items():
  tk.register_output(f"librispeech/dataset/{name}", path)

tk.register_output("librispeech/sentencepiece-2k.model", librispeech.spm_2k)
