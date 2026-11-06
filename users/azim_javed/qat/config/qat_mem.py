import sys
import os

from recipe.experiments.librispeech import run_mem
from sisyphus import tk
from i6_core.tools.compile import MakeJob, CMakeJob
from i6_core.tools.git import CloneGitRepositoryJob


# import qat.recipe.tools
async def py():
    models, results = await run_mem(filename="/u/azim.javed/experiments/training/qat/memristor_runs.txt")
