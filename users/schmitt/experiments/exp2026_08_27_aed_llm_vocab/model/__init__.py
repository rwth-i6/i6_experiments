"""
Model code for the packed-tensor Loquacious AED experiments: model definitions, train steps and
recognition. Sisyphus code lives in ``sis_recipe``.

Everything here is an own port of
:mod:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed`, behaviour-identical at
the time of the port (2026-08-27), so the packed runs can be validated against Albert's padded
reference before this setup starts diverging towards the LLM vocab.

Layout follows ``speech_llm/prefix_lm/model``:
  definitions/  -- Model + model_def
  train_steps/  -- train_def
  recognition/  -- recog_defs
"""

from .definitions.aed import Model, model_def
from .train_steps.aed_ctc_ce import train_def
from .recognition.aed_beam_search import recog_def

__all__ = ["Model", "model_def", "train_def", "recog_def"]
