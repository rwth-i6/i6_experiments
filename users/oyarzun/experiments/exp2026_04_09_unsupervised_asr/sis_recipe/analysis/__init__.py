"""Analysis tools and Sisyphus jobs for unsupervised ASR & cross-modal representations."""

import os
import importlib.util

from .distribution_analysis import (
    TextNgramDistributionAnalysisJob,
    SpeechClusterDistributionAnalysisJob,
)

_sibling_analysis_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis.py")
if os.path.exists(_sibling_analysis_py):
    _parent_pkg = __package__.rsplit(".", 1)[0]
    _spec = importlib.util.spec_from_file_location(
        f"{_parent_pkg}._pipeline_analysis", _sibling_analysis_py
    )
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__package__ = _parent_pkg
    _spec.loader.exec_module(_mod)
    analyze_encoder_states = _mod.analyze_encoder_states
    analyze_cross_attention = _mod.analyze_cross_attention
    EncoderPcaConfig = _mod.EncoderPcaConfig
    CrossAttentionConfig = _mod.CrossAttentionConfig
