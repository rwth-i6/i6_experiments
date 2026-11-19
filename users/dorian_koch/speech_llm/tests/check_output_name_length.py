"""Guard: every registered Sisyphus output path fits the filesystem's name limit.

A path component over 255 bytes cannot be created: the manager then fails to link that output on every loop
("Failed to updated output ... [Errno 36] File name too long"), so the result never appears under ``output/``
(2026-11-19: ~400 such warnings per manager start from one RL arm whose name alone was 282 characters, plus the
prefixes and suffixes its evals add).

Asserted:
  * ``too_long`` flags a component over 255 bytes (UTF-8, so non-ASCII counts by bytes) and passes one at 255;
  * with ``--graph`` (one config load, ~2 min): no output target in the real graph has such a component, with every
    alias prefix the manager would link it under.

Shown failing on the graph with the 282-character MT v6 arm name.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_output_name_length.py [--graph]
"""

import os
import sys
from pathlib import Path

SETUP = Path(__file__).absolute().parents[6]  # recipe/i6_experiments/users/dorian_koch/speech_llm/tests -> setup
NAME_MAX = 255


def too_long(path: str) -> list[str]:
    """The components of ``path`` longer than NAME_MAX bytes."""
    return [c for c in path.split("/") if len(c.encode("utf-8")) > NAME_MAX]


def check_unit():
    assert too_long("output/a/" + "x" * 255) == [], "a 255-byte component is legal"
    assert too_long("output/" + "x" * 256 + "/y") == ["x" * 256], "a 256-byte component must be flagged"
    assert too_long("output/" + "ä" * 128) == ["ä" * 128], "length counts UTF-8 bytes"
    print("[ok] too_long flags components over 255 bytes")


def check_graph():
    import logging

    os.chdir(SETUP)
    os.environ.setdefault("CUDA_HOME", "/usr")
    sys.path[:0] = ["recipe", "recipe/sisyphus"]
    logging.disable(logging.WARNING)
    from sisyphus import tk
    from sisyphus.loader import config_manager

    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
    bad, n = [], 0
    for target in tk.sis_graph.targets:
        out = getattr(target, "_output_path", None)
        if not out:
            continue
        # as ``OutputPath.run_when_done`` links it: once per alias prefix of the creating job
        creator = getattr(getattr(target, "_sis_path", None), "creator", None)
        prefixes = list(creator._sis_alias_prefixes) if creator is not None and creator._sis_alias_prefixes else [""]
        for prefix in prefixes:
            n += 1
            path = os.path.join("output", prefix, out)
            if too_long(path):
                bad.append(path)
    assert n > 100, f"only {n} output paths found: the graph walk is broken"
    assert not bad, f"{len(bad)} output paths with a component over {NAME_MAX} bytes, e.g. {bad[:3]}"
    print(f"[ok] all {n} output paths in the graph have components of at most {NAME_MAX} bytes")


if __name__ == "__main__":
    check_unit()
    if "--graph" in sys.argv:
        check_graph()
    print("check_output_name_length: ok")
