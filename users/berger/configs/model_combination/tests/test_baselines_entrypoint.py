"""Entry-point wiring checks without scheduling the historical experiment grids."""

from sisyphus import gs
import pytest

from i6_experiments.users.berger.configs import baselines
from i6_experiments.users.berger.configs.baselines import open_asr


def test_baselines_registers_open_asr_outside_corpus_contexts(monkeypatch):
    monkeypatch.setattr(gs, "ALIAS_AND_OUTPUT_SUBDIR", "test-prefix")
    calls = []
    for name in ("librispeech", "loquacious_small", "loquacious_medium", "open_asr", "qwen_tuning", "earnings_qwen", "pretrained_corrections", "speech_qwen", "controlled_comparison", "error_complementarity", "research_diagnostics"):
        monkeypatch.setattr(
            baselines,
            f"run_{name}",
            lambda name=name: calls.append((name, gs.ALIAS_AND_OUTPUT_SUBDIR)),
        )

    baselines.main()

    assert calls == [
        ("librispeech", "test-prefix/librispeech"),
        ("loquacious_small", "test-prefix/loquacious_small"),
        ("loquacious_medium", "test-prefix/loquacious_medium"),
        ("open_asr", "test-prefix"),
        ("qwen_tuning", "test-prefix"),
        ("earnings_qwen", "test-prefix"),
        ("pretrained_corrections", "test-prefix"),
        ("speech_qwen", "test-prefix"),
        ("controlled_comparison", "test-prefix"),
        ("error_complementarity", "test-prefix"),
        ("research_diagnostics", "test-prefix"),
    ]
    assert gs.ALIAS_AND_OUTPUT_SUBDIR == "test-prefix"


def test_open_asr_wrapper_uses_frozen_evaluation_matrix(monkeypatch):
    assert not hasattr(open_asr, "run_all")
    calls = []
    data = {}
    monkeypatch.setattr(
        open_asr, "run_qwen", lambda **kwargs: calls.append(kwargs) or dict(data=data, scores={}, runtimes={})
    )
    speech_scores = {"qwen-speech/native": {}}
    monkeypatch.setattr(
        open_asr, "run_qwen_speech", lambda **kwargs: calls.append(kwargs) or dict(scores=speech_scores, runtimes=speech_scores)
    )
    rnnt_calls = []
    monkeypatch.setattr(
        open_asr, "run_rnnt_qwen", lambda **kwargs: rnnt_calls.append(kwargs) or dict(scores={}, runtimes={})
    )
    open_asr.run()
    assert calls == [{"loquacious": True, "two_pass": True, "matched_controls": True}, {"data": data},
                     {"data": data, "text_size": "1.7b", "two_pass": True},
                     {"data": data, "speech_size": "1.7b"}]
    assert rnnt_calls == [
        {"data": data, "speech_baselines": {}, "tuning_only": False, "evaluation_only": True, "matched_controls": True},
        {"data": data, "speech_baselines": {}, "tuning_only": False, "evaluation_only": True, "text_size": "1.7b"},
        {"data": data, "speech_baselines": {}, "tuning_only": False, "evaluation_only": True,
         "speech_size": "1.7b", "speech_only": True},
    ]


@pytest.mark.parametrize("name", ["librispeech", "loquacious_small", "loquacious_medium"])
def test_standalone_corpus_configs_keep_alias_context(monkeypatch, name):
    import importlib

    config = importlib.import_module(f"i6_experiments.users.berger.configs.baselines.{name}")
    driver = (
        config
        if name == "librispeech"
        else importlib.import_module("i6_experiments.users.berger.configs.baselines.loquacious")
    )
    function = "run" if name == "librispeech" else f"run_{name.split('_', 1)[1]}"
    calls = []
    monkeypatch.setattr(gs, "ALIAS_AND_OUTPUT_SUBDIR", "test-prefix")
    monkeypatch.setattr(driver, function, lambda: calls.append(gs.ALIAS_AND_OUTPUT_SUBDIR))
    config.main()
    assert calls == [f"test-prefix/{name}"]
    assert gs.ALIAS_AND_OUTPUT_SUBDIR == "test-prefix"


def test_standalone_open_asr_does_not_add_context(monkeypatch):
    monkeypatch.setattr(gs, "ALIAS_AND_OUTPUT_SUBDIR", "test-prefix")
    calls = []
    monkeypatch.setattr(open_asr, "run", lambda: calls.append(gs.ALIAS_AND_OUTPUT_SUBDIR))
    open_asr.main()
    assert calls == ["test-prefix"]


def test_rnnt_entrypoint_only_registers_tuning(monkeypatch):
    from i6_experiments.users.berger.configs.baselines import rnnt_qwen
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr import rnnt_qwen as driver

    calls = []
    monkeypatch.setattr(driver, "run", lambda **kwargs: calls.append(kwargs))
    rnnt_qwen.main()
    assert calls == [{"tuning_only": True, "focused_tuning": True}]


def test_parent_import_does_not_import_corpus_drivers():
    import os
    import subprocess
    import sys

    script = """
import sys
import i6_experiments.users.berger.configs.baselines
prefix = 'i6_experiments.users.berger.configs.baselines.'
assert not any(prefix + name in sys.modules for name in ('librispeech', 'loquacious', 'open_asr'))
"""
    subprocess.run([sys.executable, "-c", script], env=os.environ.copy(), check=True)


@pytest.mark.parametrize(
    "name",
    ["", "librispeech", "loquacious_small", "loquacious_medium", "open_asr", "rnnt_qwen", "wordwise_lm", "results"],
)
def test_filename_entrypoints_have_py(monkeypatch, name):
    import importlib

    module = importlib.import_module("i6_experiments.users.berger.configs.baselines" + ("." + name if name else ""))
    calls = []
    if name:
        monkeypatch.setattr(module, "py", lambda: calls.append("py"))
        module.main()
        assert calls == ["py"]
    else:
        monkeypatch.setattr(module, "main", lambda: calls.append("main"))
        module.py()
        assert calls == ["main"]
