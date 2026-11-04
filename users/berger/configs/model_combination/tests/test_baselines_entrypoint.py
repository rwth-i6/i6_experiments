"""Entry-point wiring checks without scheduling the historical experiment grids."""


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
