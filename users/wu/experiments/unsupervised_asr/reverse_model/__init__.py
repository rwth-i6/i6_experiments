"""Reverse-model (phi) initialisations of phase 4a: the given duration prior, the phi-first EM fit
(L2-1 stage 1) with its label-free genmarg selection, the 10 h supervised reverse init and the p0
recognizer (analysis only), and the L2-0 competence ladder (analysis only).

Sisyphus-side modules: ``duration_prior``, ``phi_first``, ``genmarg``, ``supervised``,
``supervised_data``, ``p0``, ``ladder``.
RETURNN-side modules (imported by the written configs only): ``genmarg_steps``, ``genmarg_decode``,
``supervised_steps``, ``p0_steps``.
"""
