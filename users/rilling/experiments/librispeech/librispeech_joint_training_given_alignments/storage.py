from typing import Dict
from dataclasses import dataclass
from sisyphus import tk
from .pytorch_networks.shared.configs import ModelConfigV1

synthetic_ogg_zip_data = {}

def add_ogg_zip(name: str, ogg_zip: tk.Path):
    global synthetic_ogg_zip_data
    synthetic_ogg_zip_data[name] = ogg_zip
    
duration_alignments = {}

def add_duration(name: str, duration_hdf: tk.Path):
    global duration_alignments
    duration_alignments[name] = duration_hdf

@dataclass
class TTSModel:
    config: ModelConfigV1
    checkpoint: tk.Path

tts_models: Dict[str, TTSModel] = {}

def add_tts_model(name: str, model: TTSModel):
    assert name not in tts_models.keys(), "A model with that name is already stored!"
    tts_models[name] = model