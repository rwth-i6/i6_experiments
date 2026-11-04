"""Historical imports must resolve to moved code without renaming job classes."""

import importlib

from i6_experiments.users.berger.configs.model_combination import bootstrap


def test_finder_is_idempotent():
    before = list(__import__("sys").meta_path)
    bootstrap.activate()
    assert __import__("sys").meta_path == before


def test_split_imports_reuse_historical_classes():
    for new, old in (
        ("i6_experiments.users.berger.data.huggingface", bootstrap.PREFIX + ".data.huggingface"),
        (
            "i6_experiments.users.berger.model_pipelines.common.rnnt_llm",
            bootstrap.PREFIX + ".model_pipelines.common.rnnt_llm",
        ),
        (
            "i6_experiments.users.berger.configs.model_combination.experiments.open_asr",
            bootstrap.PREFIX + ".experiments.open_asr",
        ),
    ):
        original = importlib.import_module(old)
        spec = original.__spec__
        assert importlib.import_module(new) is original
        assert original.__spec__ is spec
