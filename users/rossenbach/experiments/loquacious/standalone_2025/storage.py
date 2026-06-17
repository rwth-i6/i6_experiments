"""
Data to store globally to make it easier to transfer arbitrarily between the different experiments
"""
from dataclasses import dataclass
from sisyphus import tk
from typing import Any, Dict, List

from i6_core.returnn.config import ReturnnConfig

from .pipeline import ASRModel, NeuralLM



# Neural LM Models -------------------------------------------------------------------------------------------------------

_lm_models: Dict[str, tk.Path] = {}

def add_lm(name: str, lm_model: NeuralLM):
    global _lm_models
    assert name not in _lm_models.keys()
    _lm_models[name] = lm_model


def get_lm_model(name: str) -> NeuralLM:
    global _lm_models
    return _lm_models[name]


# Vocoder Models -------------------------------------------------------------------------------------------------------

@dataclass
class VocoderPackage:
    checkpoint: tk.Path
    config: Dict[str, Any]

vocoders: Dict[str, VocoderPackage] = {}

def add_vocoder(name: str, vocoder: VocoderPackage):
    global vocoders
    assert name not in vocoders.keys()
    vocoders[name] = vocoder


# Alignments -----------------------------------------------------------------------------------------------------------

duration_alignments = {}

def add_duration(name: str, duration_hdf: tk.Path):
    global duration_alignments
    assert name not in duration_alignments.keys()
    duration_alignments[name] = duration_hdf

# GMM Alignments -------------------------------------------------------------------------------------------------------

label_alignments = {}

def add_label_alignment(name: str, label_hdfs: List[tk.Path]):
    global label_alignments
    assert name not in label_alignments.keys()
    label_alignments[name] = label_hdfs

# Synthetic data -------------------------------------------------------------------------------------------------------

synthetic_ogg_zip_data = {}
synthetic_bliss_data = {}
synthetic_data_lexica = {}


def add_synthetic_data(name: str, ogg_zip: tk.Path, bliss: tk.Path):
    global synthetic_ogg_zip_data
    global synthetic_bliss_data
    assert name not in synthetic_ogg_zip_data.keys()
    synthetic_ogg_zip_data[name] = ogg_zip
    synthetic_bliss_data[name] = bliss

def get_synthetic_data(name: str):
    global synthetic_ogg_zip_data
    global synthetic_bliss_data
    assert name in synthetic_ogg_zip_data.keys()
    return synthetic_bliss_data[name], synthetic_ogg_zip_data[name]

def add_synthetic_data_lexicon(name: str, lexicon: tk.Path):
    global synthetic_data_lexica
    assert name not in synthetic_data_lexica.keys()
    synthetic_data_lexica[name] = lexicon

def get_synthetic_data_lexicon(name: str):
    global synthetic_data_lexica
    assert name in synthetic_data_lexica.keys()
    return synthetic_data_lexica[name]

