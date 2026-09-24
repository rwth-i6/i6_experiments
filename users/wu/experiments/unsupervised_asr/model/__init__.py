"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/ and src/speech_llm/prefix_lm/model/
(package marker; each module names its own source file).

The RETURNN-side training closure of the unsupervised-ASR runs: theta (``recognizer``), phi
(``reverse``), the blank-free lattice and its terms, the models and their train steps.  Nothing is
imported here, so importing one submodule never pulls in torch-heavy or k2-only siblings.
"""
