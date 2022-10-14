"""
Sisyphus main entry
"""

import os
import importlib


__my_dir__ = os.path.dirname(os.path.abspath(__file__))


def sis_config_main():
    """sis config function"""
    prefix = "exp_fs_base/"
    for filename in sorted(os.listdir(__my_dir__)):
        if not filename.endswith(".py"):
            continue
        if filename.startswith("_"):
            continue
        basename = filename[:-len('.py')]
        name = f"{__package__}.{basename}"
        mod = importlib.import_module(name)
        if not hasattr(mod, "sis_run_with_prefix"):
            continue
        mod.sis_run_with_prefix(prefix + basename)


py = sis_config_main  # `py` is the default sis config function name
