"""
Data to store globally to make it easier to transfer arbitrarily between the different experiments
"""
from dataclasses import dataclass
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn.config import ReturnnConfig

from .pipeline import ASRModel, NeuralLM


# CTC Models for RNN-T init --------------------------------------------------------------------------------------------

_ctc_models: Dict[str, tk.Path] = {}

def add_ctc_model(name: str, asr_model: ASRModel):
    global _ctc_models
    assert name not in _ctc_models.keys()
    _ctc_models[name] = asr_model


def get_ctc_model(name: str) -> ASRModel:
    global _ctc_models
    return _ctc_models[name]


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


# Synthetic data -------------------------------------------------------------------------------------------------------

synthetic_ogg_zip_data = {}
synthetic_bliss_data = {}


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

