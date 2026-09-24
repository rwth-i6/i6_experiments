"""Shared pytest markers of the unsupervised_asr test suite (test plan 2026-09-24, section 0).

* ``slow``     -- more than ~30 s on a CPU; deselect with ``-m "not slow"``.
* ``gpu``      -- needs CUDA; skipped when ``torch.cuda.is_available()`` is False.
* ``k2``       -- needs the ``k2`` module; skipped when it cannot be imported.
* ``artefact`` -- reads banked outputs; skipped unless ``SAE_ARTEFACT_DIR`` is set.

The package itself is made importable the way the suite always was: by the caller's
``PYTHONPATH`` (the setup's ``recipe/``, ``recipe/returnn`` and ``sisyphus/``). Nothing here touches
``sys.path``.
"""

import importlib.util
import os

import pytest

_MARKERS = {
    "slow": "more than ~30 s on a CPU (deselect with -m 'not slow')",
    "gpu": "needs CUDA; skipped without it",
    "k2": "needs the k2 module; skipped without it",
    "artefact": "needs banked outputs; skipped unless SAE_ARTEFACT_DIR is set",
}


def pytest_configure(config):
    for name, text in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _k2_importable() -> bool:
    """``pytest.importorskip("k2")`` semantics: installed is not enough, the import must succeed."""
    if importlib.util.find_spec("k2") is None:
        return False
    try:
        importlib.import_module("k2")
    except Exception:  # an installed but broken k2 (ABI mismatch) is skipped like a missing one
        return False
    return True


def pytest_collection_modifyitems(config, items):
    cuda = None
    has_k2 = None
    artefact_dir = os.environ.get("SAE_ARTEFACT_DIR")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            if cuda is None:
                cuda = _cuda_available()
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="gpu: CUDA is not available"))
        if item.get_closest_marker("k2") is not None:
            if has_k2 is None:
                has_k2 = _k2_importable()
            if not has_k2:
                item.add_marker(pytest.mark.skip(reason="k2: the k2 module cannot be imported"))
        if item.get_closest_marker("artefact") is not None and not artefact_dir:
            item.add_marker(pytest.mark.skip(reason="artefact: SAE_ARTEFACT_DIR is not set"))
