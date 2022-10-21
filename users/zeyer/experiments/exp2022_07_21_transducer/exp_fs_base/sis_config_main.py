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
        # Read the file and search for expected content.
        # This is much faster than importing the file,
        # so good to exclude files which are not relevant.
        file_content = open(filename).read().splitlines()
        found_entry_function = False
        found_exclude = False
        for line in file_content[:30]:
            if not found_entry_function and line.startswith("def sis_run_with_prefix("):
                found_entry_function = True
                continue
            if line.startswith("_exclude_me = True"):
                found_exclude = True
                break
        if found_exclude:
            continue
        if not found_entry_function:
            continue
        # Ok import it.
        basename = filename[:-len('.py')]
        name = f"{__package__}.{basename}"
        mod = importlib.import_module(name)
        assert hasattr(mod, "sis_run_with_prefix")
        mod.sis_run_with_prefix(prefix + basename)


py = sis_config_main  # `py` is the default sis config function name
