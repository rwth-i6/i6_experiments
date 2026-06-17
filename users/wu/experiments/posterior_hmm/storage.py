"""
Data to store globally to make it easier to transfer arbitrarily between the different experiments
"""
from dataclasses import dataclass
from sisyphus import tk
from typing import Any, Dict

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
    if name in _lm_models:
        # Multiple config entry points can "ensure" the same shared LM in one graph
        # (e.g. recognition and PPL). Treat identical registrations as idempotent.
        assert repr(_lm_models[name]) == repr(lm_model), (
            f"LM model {name!r} already registered with a different value:\n"
            f"existing: {_lm_models[name]!r}\nnew: {lm_model!r}"
        )
        return
    _lm_models[name] = lm_model


def get_lm_model(name: str) -> NeuralLM:
    global _lm_models
    return _lm_models[name]
