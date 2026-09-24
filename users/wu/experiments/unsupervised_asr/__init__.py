"""Phase 4a of the unsupervised ASR (SAE) campaign: the exact-marginal cycle (EMC) setup.

Sisyphus jobs and RETURNN code that build every input from public raw sources and train / read the
phase-4a arms.  Layout, quick start and pins: README.md.

Kept import-free: the k2 child processes run ``python -m <this package>.lm.lexlat_k2_official`` and
``<this package>.model.lexlat_k2`` under the k2 interpreter, which has no sisyphus.
"""
