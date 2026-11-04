"""Historical imports must resolve to moved code without renaming job classes."""

import hashlib
import importlib
import json
import pickle
from pathlib import Path

from i6_experiments.users.berger.configs.model_combination import bootstrap


def test_research_source_manifest_is_complete_and_unchanged():
    root = Path(bootstrap.__file__).parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    repo = bootstrap.REPOSITORY
    for row in manifest["files"]:
        if row["research_path"] in manifest["relocation_exceptions"]:
            continue
        source = repo / row["research_path"]
        assert source.is_file(), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"], source


def test_legacy_job_classes_keep_module_and_pickle_identity():
    name = bootstrap.PREFIX + ".model_pipelines.common.rnnt_llm"
    module = importlib.import_module(name)
    assert Path(module.__file__).is_relative_to(bootstrap.RESEARCH)
    assert module.RnntQwenRecogJob.__module__ == name
    assert pickle.loads(pickle.dumps(module.RnntQwenRecogJob)) is module.RnntQwenRecogJob


def test_unchanged_template_modules_are_reused():
    module = importlib.import_module(bootstrap.PREFIX + ".model_pipelines.ctc.pytorch_modules")
    assert Path(module.__file__).is_relative_to(bootstrap.TEMPLATES)


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


def test_historical_entrypoints_load_the_canonical_files():
    module = importlib.import_module(bootstrap.LEGACY_CONFIGS + ".rnnt_qwen")
    assert Path(module.__file__) == bootstrap.CONFIGS / "rnnt_qwen.py"
