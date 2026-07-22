import sys
import os

from recipe.experiments.librispeech import run_debug
from sisyphus import tk

def py():
    models, results = run_debug(filename="/u/azim.javed/experiments/training/qat/report_debug.txt")
