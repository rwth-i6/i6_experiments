"""Explicit local assets: optional by default, errors when an asset was requested."""

import os
from pathlib import Path

import pytest


def requested_asset(variable, pattern=None):
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"Set {variable} to enable this integration check")
    path = Path(value).resolve()
    if not path.is_dir() or (pattern and not list(path.glob(pattern))):
        pytest.fail(f"Explicit {variable} is not a usable asset directory: {path}")
    return path


@pytest.fixture
def rasr_lib_dir():
    return requested_asset("RASR_LIB_DIR", "librasr*.so")


@pytest.fixture
def granite_tokenizer_dir():
    return requested_asset("GRANITE_TOKENIZER_DIR")


@pytest.fixture
def rasr_dynamic_lib_dir():
    return requested_asset("RASR_DYNAMIC_LIB_DIR", "librasr*.so")


@pytest.fixture
def rasr_bpe_lib_dir():
    return requested_asset("RASR_BPE_LIB_DIR", "librasr*.so")
