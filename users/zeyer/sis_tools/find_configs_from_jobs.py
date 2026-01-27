#!/usr/bin/env python3

"""
From job directories, find and list all unique Sisyphus configuration files used,
based on the Sisyphus stacktrace in the info file.
"""

import os
import re
import sys
import argparse
from collections import Counter
from functools import reduce
from typing import TypeVar

_my_dir = os.path.dirname(__file__)
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = _setup_base_dir + "/tools/sisyphus"
_returnn_dir = _setup_base_dir + "/tools/returnn"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        if _returnn_dir not in sys.path:
            sys.path.append(_returnn_dir)


_setup()


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("dirs", nargs="+", help="Directory containing job directories")
    args = arg_parser.parse_args()

    state = _State()
    for base_dir in args.dirs:
        if _is_job_dir(base_dir):
            _check_job(base_dir, state)
            continue
        for dirpath, dirnames, filenames in os.walk(base_dir):
            job_dirnames = set()
            for dirname in dirnames:
                job_dir = f"{dirpath}/{dirname}"
                if _is_job_dir(job_dir):
                    job_dirnames.add(dirname)
                    _check_job(job_dir, state)
            dirnames[:] = [d for d in dirnames if d not in job_dirnames]

    if state.job_count == 0:
        print("No job directories found.", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"Processed {state.job_count} job directories.", file=sys.stderr)
    print("Unique configuration files used (count):")
    for config_file, count in state.configs.most_common():
        print(f"{count:6d}  {config_file}")


# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
# 7-bit C1 ANSI sequences
_AnsiEscapeRe = re.compile(
    r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
    """,
    re.VERBOSE,
)

# Example stacktrace line (among the locals):
#   File "/.../tools/sisyphus/sisyphus/loader.py", line 90, in ConfigManager.load_config_file
_StacktraceEntryRe = re.compile(r'File "(.+)", line (\d+), in (.+)')


class _State:
    def __init__(self):
        self.job_count = 0
        self.configs = Counter()


def _check_job(job_dir: str, state: _State):
    state.job_count += 1
    info_file = f"{job_dir}/info"
    with open(info_file, "r") as f:
        info_content = f.read()
    match = re.search(r"STACKTRACE:\n(.*)", info_content, re.DOTALL)
    if not match:
        print("WARNING: No stacktrace found in info file:", info_file, file=sys.stderr)
        return
    stacktrace = match.group(1)

    # Sometimes the job is created in multiple places, and then there are multiple stacktraces.
    if "\nSTACKTRACE:\n" in stacktrace:
        stacktrace = stacktrace.split("\nSTACKTRACE:\n", 1)[0]

    stacktrace = _AnsiEscapeRe.sub("", stacktrace)
    stack = []
    for line in stacktrace.splitlines():
        line = line.strip()
        match = _StacktraceEntryRe.match(line)
        if not match:
            continue
        stack.append((match.group(1), int(match.group(2)), match.group(3)))
    assert stack, f"no stacktrace entries found in job {job_dir}"
    idxs = [
        i
        for i, (filename, _, func_name) in enumerate(stack)
        if func_name == "ConfigManager.load_config_file" and filename.endswith("/sisyphus/sisyphus/loader.py")
    ]
    assert idxs, f"no ConfigManager.load_config_file entry found in job {job_dir}"
    assert len(idxs) == 1, f"multiple ConfigManager.load_config_file entries found in job {job_dir}"
    assert idxs[0] + 1 < len(stack), f"no caller entry found after ConfigManager.load_config_file in job {job_dir}"
    caller_entry = stack[idxs[0] + 1]
    config_file = caller_entry[0]
    state.configs[config_file] += 1

    sys.stderr.write(".")
    sys.stderr.flush()


def _is_job_dir(job_dir: str) -> bool:
    """
    :param job_dir: absolute or relative path to job directory
    """
    base_name = os.path.basename(job_dir)
    if "." not in base_name:
        return False
    if not os.path.isfile(job_dir + "/info"):
        return False
    if not os.path.isdir(job_dir + "/output"):
        return False
    return True


if __name__ == "__main__":
    main()
