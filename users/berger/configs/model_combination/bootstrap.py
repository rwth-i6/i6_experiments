"""Resolve historical job modules from relocated research source files.

No classes are renamed, no pickle contents rewritten and no hash is overridden.
This branch activates the finder at package import so generated RETURNN configs
and Sisyphus worker processes use the same definitions as the manager. Unchanged
modules fall back to the template files. The template branch has no bootstrap.
"""

import importlib.abc
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

PREFIX = "i6_experiments.example_setups.seq2seq_rasr_2025"
LEGACY_CONFIGS = "i6_experiments.users.berger.configs.baselines"
CONFIGS = Path(__file__).resolve().parent
RESEARCH = CONFIGS.parents[1]  # users/berger
REPOSITORY = CONFIGS.parents[3]
TEMPLATES = REPOSITORY / "example_setups/seq2seq_rasr_2025"


class AliasLoader(importlib.abc.Loader):
    """Expose the same module at its new path without executing it twice."""

    def __init__(self, original_name):
        self.original_name = original_name

    def create_module(self, spec):
        module = importlib.import_module(self.original_name)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery temporarily sets the alias spec on the shared object.
        module.__spec__ = self.original_spec


def research_path(relative):
    """Map an old setup-relative name to the split source layout."""
    parts = Path(relative).parts
    if parts and parts[0] in {"data", "model_pipelines"}:
        return RESEARCH.joinpath(*parts)
    return CONFIGS.joinpath(*parts)


def source_path(relative):
    """Resolve files for source fingerprints without duplicating the sources."""
    source = research_path(relative)
    return source if source.exists() else TEMPLATES / relative


class ResearchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        aliases = {
            "i6_experiments.users.berger.data": PREFIX + ".data",
            "i6_experiments.users.berger.model_pipelines": PREFIX + ".model_pipelines",
            "i6_experiments.users.berger.configs.model_combination.experiments": PREFIX + ".experiments",
            "i6_experiments.users.berger.configs.model_combination.tools": PREFIX + ".tools",
        }
        for canonical, historical in aliases.items():
            if fullname == canonical or fullname.startswith(canonical + "."):
                original = historical + fullname[len(canonical) :]
                suffix = original[len(PREFIX) + 1 :].replace(".", "/")
                is_package = research_path(suffix).is_dir() or (TEMPLATES / suffix).is_dir()
                return importlib.util.spec_from_loader(fullname, AliasLoader(original), is_package=is_package)
        if fullname == LEGACY_CONFIGS:
            return importlib.util.spec_from_file_location(
                fullname,
                CONFIGS / "all.py",
                submodule_search_locations=[str(CONFIGS), str(RESEARCH / "configs/baselines")],
            )
        if fullname.startswith(LEGACY_CONFIGS + "."):
            suffix = fullname[len(LEGACY_CONFIGS) + 1 :].replace(".", "/")
            locations = [CONFIGS / suffix, RESEARCH / "configs/baselines" / suffix]
        elif fullname == PREFIX:
            locations = [TEMPLATES]
        elif fullname.startswith(PREFIX + "."):
            suffix = fullname[len(PREFIX) + 1 :].replace(".", "/")
            locations = [research_path(suffix), TEMPLATES / suffix]
        else:
            return None
        directories = [str(p) for p in locations if p.is_dir()]
        if directories:
            init = next((p / "__init__.py" for p in locations if (p / "__init__.py").is_file()), None)
            if init is not None:
                return importlib.util.spec_from_file_location(fullname, init, submodule_search_locations=directories)
        for location in locations:
            source = location.with_suffix(".py")
            if source.is_file():
                return importlib.util.spec_from_file_location(fullname, source)
        if directories:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = directories
            return spec
        return None


def activate():
    if any(isinstance(f, ResearchFinder) for f in sys.meta_path):
        return
    loaded = [name for name in sys.modules if name == PREFIX or name.startswith(PREFIX + ".")]
    if loaded:
        raise RuntimeError("Activate ASR research before importing templates; restart this Python process")
    sys.meta_path.insert(0, ResearchFinder())
