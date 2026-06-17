"""
Pure-Python tests for the synthetic training data pipeline.

Run from the setup root:
    cd /home/tt201262/setups/2026-01-speech-llm
    PYTHONPATH=recipe .venv/bin/python -m pytest \
        recipe/i6_experiments/users/dorian_koch/speech_llm/test_pipeline.py -v

No GPU or heavy venv (chatterbox/moshi) required.
"""

import importlib.machinery
import json
import os
import random
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Auto-mock strategy: intercept imports at the meta-path level (PEP 451).
#
# Rules (applied in order):
#  1. Always mock: the listed heavy namespaces (torch, i6_core, …)
#  2. Never mock: i6_experiments itself, i6_experiments.users,
#                 i6_experiments.users.dorian_koch and everything under it
#                 (those are the modules we actually want to test).
#  3. Mock everything else under i6_experiments.* so that other users'
#     recipe code (zeyer, common, …) never executes.
# ---------------------------------------------------------------------------

_ALWAYS_MOCK = (
    "sisyphus",
    "i6_core",
    "openai",
    "chatterbox",
    "torch",
    "torchaudio",
    "soundfile",
    "returnn",
)

_DORIAN_PREFIX = "i6_experiments.users.dorian_koch"
_REAL_I6_ANCESTORS = {"i6_experiments", "i6_experiments.users"}


def _should_mock(fullname: str) -> bool:
    if any(fullname == p or fullname.startswith(p + ".") for p in _ALWAYS_MOCK):
        return True
    # Keep the ancestor packages and dorian_koch real
    if fullname in _REAL_I6_ANCESTORS:
        return False
    if fullname == _DORIAN_PREFIX or fullname.startswith(_DORIAN_PREFIX + "."):
        return False
    # Mock everything else under i6_experiments (other users, common, …)
    if fullname.startswith("i6_experiments."):
        return True
    return False


class _AutoMockFinder:
    """Return an empty mock package for imports matching _should_mock()."""

    def find_spec(self, fullname, path, target=None):
        if _should_mock(fullname):
            return importlib.machinery.ModuleSpec(fullname, loader=self, is_package=True)
        return None

    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.parent or spec.name
        mod.__all__ = []
        return mod

    def exec_module(self, module):
        def _getattr(attr):
            val = MagicMock()
            setattr(module, attr, val)
            return val

        module.__getattr__ = _getattr


sys.meta_path.insert(0, _AutoMockFinder())

# Give sisyphus.Job a real base class so subclasses can inherit from it.
import importlib as _il

_sis = _il.import_module("sisyphus")
_sis.Job = type(
    "Job",
    (),
    {
        "output_path": lambda self, *a, **k: MagicMock(),
        "output_var": lambda self, *a, **k: MagicMock(),
    },
)
_sis.Task = MagicMock()
_sis.tk = MagicMock()

# datasets: use the real package if available (it's in the setup .venv).
try:
    import datasets as _real_ds  # noqa: F401
except ImportError:
    _ds = types.ModuleType("datasets")
    _ds.__path__ = []
    for _attr in [
        "Dataset",
        "DatasetDict",
        "Features",
        "Value",
        "Sequence",
        "Audio",
        "List",
        "load_dataset",
        "load_from_disk",
        "concatenate_datasets",
    ]:
        setattr(_ds, _attr, MagicMock())
    sys.modules["datasets"] = _ds


# ---------------------------------------------------------------------------
# hf_to_dialogue: template names and _pick_template
# ---------------------------------------------------------------------------


def test_template_names_length():
    from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import (
        DIALOGUE_INSTRUCTION_TEMPLATES,
        DIALOGUE_INSTRUCTION_TEMPLATE_NAMES,
    )

    assert len(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES) == len(DIALOGUE_INSTRUCTION_TEMPLATES)


def test_template_names_unique():
    from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import (
        DIALOGUE_INSTRUCTION_TEMPLATE_NAMES,
    )

    assert len(set(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES)) == len(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES)


def test_pick_template_determinism():
    """Same uid always returns the same (template_text, template_name) pair."""
    from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import _pick_template

    for uid in ["42", "hello", "question_id_999", ""]:
        t1, n1 = _pick_template(uid)
        t2, n2 = _pick_template(uid)
        assert t1 is t2, f"Template text should be identical object for uid={uid!r}"
        assert n1 == n2, f"Template name should be stable for uid={uid!r}"


def test_pick_template_spread():
    """Different uids should reach every template index."""
    from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import (
        DIALOGUE_INSTRUCTION_TEMPLATES,
        _pick_template,
    )

    n = len(DIALOGUE_INSTRUCTION_TEMPLATES)
    seen_names = set()
    for i in range(n * 100):
        _, name = _pick_template(str(i))
        seen_names.add(name)
    assert len(seen_names) == n, f"Expected all {n} templates to be selected; got {len(seen_names)}: {seen_names}"


def test_dialogue_schema_rejects_empty_text():
    """The guided_json schema must reject turns with empty text."""
    from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import _DIALOGUE_JSON_SCHEMA

    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")

    valid = [{"speaker": "user", "text": "Hello"}, {"speaker": "assistant", "text": "Hi"}]
    jsonschema.validate(valid, _DIALOGUE_JSON_SCHEMA)  # must not raise

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            [{"speaker": "user", "text": ""}, {"speaker": "assistant", "text": "Hi"}],
            _DIALOGUE_JSON_SCHEMA,
        )


# ---------------------------------------------------------------------------
# chatterbox_inference: available_speakers is sorted
# ---------------------------------------------------------------------------


def test_available_speakers_sorted():
    """available_speakers must return a sorted list for RNG reproducibility."""
    from i6_experiments.users.dorian_koch.speech_llm.chatterbox_inference import (
        available_speakers,
    )

    unsorted_files = ["z_voice.wav", "a_voice.wav", "m_voice.wav", "not_wav.mp3"]
    with patch("os.listdir", return_value=unsorted_files):
        result = available_speakers("/fake/dir")
    assert result == sorted(result), "available_speakers result must be sorted"
    assert "not_wav" not in result, "Non-.wav files must be excluded"
    assert result == ["a_voice", "m_voice", "z_voice"]


# ---------------------------------------------------------------------------
# Silence sampler: bounds
# ---------------------------------------------------------------------------


def test_silence_sampler_bounds():
    """All sampled silences must lie within [-0.3, 0.6]."""
    random.seed(42)

    def silence_length_sampler():
        val = random.gauss(0.2, 0.4)
        while val < -0.3 or val > 0.6:
            val = random.gauss(0.2, 0.4)
        return val

    samples = [silence_length_sampler() for _ in range(2000)]
    assert all(-0.3 <= s <= 0.6 for s in samples), "Silence sampler produced out-of-range value"


# ---------------------------------------------------------------------------
# HfDialogueCleaner: backtick stripping (logic tested inline)
# ---------------------------------------------------------------------------


def _clean_dialogue_str(s: str) -> str:
    """Replicate HfDialogueCleaner.clean_dialogue's strip logic."""
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json") :]
    if s.endswith("```"):
        s = s[: -len("```")]
    return s.strip()


def test_cleaner_strips_markdown_backticks():
    payload = [{"speaker": "user", "text": "Hello"}]
    wrapped = "```json\n" + json.dumps(payload) + "\n```"
    assert json.loads(_clean_dialogue_str(wrapped)) == payload


def test_cleaner_passes_plain_json():
    payload = [{"speaker": "user", "text": "A?"}, {"speaker": "assistant", "text": "B."}]
    assert json.loads(_clean_dialogue_str(json.dumps(payload))) == payload


# ---------------------------------------------------------------------------
# moshi_arrow_dataset: _norm_aligns (skip if finetune deps missing)
# ---------------------------------------------------------------------------


def _try_import_norm_aligns():
    try:
        from i6_experiments.users.dorian_koch.speech_llm.moshi_arrow_dataset import _norm_aligns

        return _norm_aligns
    except (ImportError, Exception):
        return None


def test_norm_aligns_struct_format():
    fn = _try_import_norm_aligns()
    if fn is None:
        pytest.skip("finetune (moshi-finetune fork) not on PYTHONPATH")

    result = fn(
        [
            {"text": "hello", "start": 0.1, "end": 0.5, "speaker": "SPEAKER_MAIN"},
            {"text": "world", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_MAIN"},
        ]
    )
    assert result == [("hello", (0.1, 0.5), "SPEAKER_MAIN"), ("world", (0.6, 1.0), "SPEAKER_MAIN")]


def test_norm_aligns_legacy_format():
    fn = _try_import_norm_aligns()
    if fn is None:
        pytest.skip("finetune (moshi-finetune fork) not on PYTHONPATH")

    result = fn([["hello", [0.1, 0.5], "SPEAKER_MAIN"], ["world", [0.6, 1.0], "SPEAKER_MAIN"]])
    assert result == [("hello", (0.1, 0.5), "SPEAKER_MAIN"), ("world", (0.6, 1.0), "SPEAKER_MAIN")]


def test_norm_aligns_empty():
    fn = _try_import_norm_aligns()
    if fn is None:
        pytest.skip("finetune (moshi-finetune fork) not on PYTHONPATH")
    assert fn([]) == []


# ---------------------------------------------------------------------------
# common.py: shared subprocess/server helpers (added with the de-dup refactor)
# ---------------------------------------------------------------------------


def test_pick_free_port_is_bindable():
    """pick_free_port returns an int near the base that is actually bindable."""
    import socket

    os.environ.pop("SLURM_JOB_ID", None)
    from i6_experiments.users.dorian_koch.speech_llm.common import pick_free_port

    p = pick_free_port(18000)
    assert isinstance(p, int)
    assert 18000 <= p <= 18000 + 999 + 50
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", p))
    s.close()


def test_run_worker_script_builds_cmd_and_env():
    """run_worker_script stringifies argv and applies PYTHONUNBUFFERED + extra_env,
    and omits HF_HOME when with_hf_home=False."""
    from i6_experiments.users.dorian_koch.speech_llm import common

    captured = {}

    def fake_run(cmd, env=None, check=None):
        captured["cmd"] = cmd
        captured["env"] = env

    with patch.object(common.subprocess, "run", fake_run):
        common.run_worker_script(
            "py",
            "/s.py",
            ["--a", 1, "--b", "x"],
            log_label="unit",
            with_hf_home=False,
            extra_env={"FOO": "bar"},
        )
    assert captured["cmd"] == ["py", "/s.py", "--a", "1", "--b", "x"]
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["env"]["FOO"] == "bar"
    assert "HF_HOME" not in captured["env"]
