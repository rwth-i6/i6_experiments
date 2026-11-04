"""Resolve historical job modules from relocated research source files.

No classes are renamed, no pickle contents rewritten and no hash is overridden.
This branch activates the finder at package import so generated RETURNN configs
and Sisyphus worker processes use the same definitions as the manager. Unchanged
modules fall back to the template files. The template branch has no bootstrap.
"""

import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

PREFIX = "i6_experiments.example_setups.seq2seq_rasr_2025"
RESEARCH = Path(__file__).resolve().parent / "setup"
TEMPLATES = Path(__file__).resolve().parents[3] / "example_setups/seq2seq_rasr_2025"


class ResearchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != PREFIX and not fullname.startswith(PREFIX + "."):
            return None
        suffix = fullname[len(PREFIX) :].lstrip(".").replace(".", "/")
        locations = [RESEARCH / suffix, TEMPLATES / suffix]
        directories = [str(p) for p in locations if p.is_dir()]
        if directories:
            init = next((p / "__init__.py" for p in locations if (p / "__init__.py").is_file()), None)
            if init is not None:
                return importlib.util.spec_from_file_location(fullname, init, submodule_search_locations=directories)
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = directories
            return spec
        for location in locations:
            source = location.with_suffix(".py")
            if source.is_file():
                return importlib.util.spec_from_file_location(fullname, source)
        return None


def activate():
    if any(isinstance(f, ResearchFinder) for f in sys.meta_path):
        return
    loaded = [name for name in sys.modules if name == PREFIX or name.startswith(PREFIX + ".")]
    if loaded:
        raise RuntimeError("Activate ASR research before importing templates; restart this Python process")
    sys.meta_path.insert(0, ResearchFinder())
