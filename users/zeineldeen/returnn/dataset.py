__all__ = ['ExtractDatasetStatisticsJob']

from sisyphus import *

import numpy
import os
import shutil
import subprocess

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob
from i6_core.util import create_executable


class ExtractDatasetStatisticsJob(ExtractDatasetMeanStddevJob):
    """
    Alias for old pipelines
    """
    pass