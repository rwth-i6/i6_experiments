#!/usr/bin/env python3
"""Runs the setup root's ``read_benchmarks.py``, the one benchmark ledger (per-grader tables, ``spoke`` / ``acc|spoke``).

This used to be an older copy that read only the pre-grader flat layout and ranked tags across graders in one table.

Run from the setup root:  python3 read_benchmarks.py [-q] [substring ...]
"""

import os
import runpy
import sys

ledger = os.path.join(os.getcwd(), "read_benchmarks.py")
if not os.path.isfile(ledger):
    sys.exit(f"run this from the setup root ({ledger} not found)")
runpy.run_path(ledger, run_name="__main__")
