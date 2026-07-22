import sys
import os

from recipe.experiments.librispeech import run_all
from sisyphus import tk

def py():
    models, results = run_all(filename="/u/azim.javed/experiments/training/qat/report.txt")
