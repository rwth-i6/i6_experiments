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
_ctc_alignment: Dict[str, tk.Path] = {}
_rnnt_hypos: Dict[str, tk.Path] = {}

def add_ctc_model(name: str, asr_model: ASRModel):
    global _ctc_models
    assert name not in _ctc_models.keys()
    _ctc_models[name] = asr_model


def get_ctc_model(name: str) -> ASRModel:
    global _ctc_models
    return _ctc_models[name]


def add_ctc_forced_alignment(name: str, alignment_path: tk.Path):
    global _ctc_alignment
    assert name not in _ctc_alignment.keys()
    _ctc_alignment[name] = alignment_path


def get_ctc_forced_alignment(name: str) -> tk.Path:
    global _ctc_alignment
    return _ctc_alignment[name]


def add_rnnt_hypo(name: str, hypo_path: tk.Path):
    global _rnnt_hypos
    assert name not in _rnnt_hypos.keys()
    _rnnt_hypos[name] = hypo_path


def get_rnnt_hypo(name: str) -> tk.Path:
    global _rnnt_hypos
    return _rnnt_hypos[name]

# Neural LM Models --------------------------------------------------------------------------------------------

_lm_models: Dict[str, tk.Path] = {}


def add_lm(name: str, lm_model: NeuralLM):
    global _lm_models
    assert name not in _lm_models.keys()
    _lm_models[name] = lm_model


def get_lm_model(name: str) -> NeuralLM:
    global _lm_models
    return _lm_models[name]