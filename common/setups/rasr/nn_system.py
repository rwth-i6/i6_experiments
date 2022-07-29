__all__ = ["NnSystem"]

import copy
import itertools
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath, MultiOutputPath

from .rasr_system import RasrSystem

from .util import (
    RasrInitArgs,
    ReturnnRasrDataInput,
    OggZipHdfDataInput,
    HybridArgs,
    NnRecogArgs,
    RasrSteps,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class NnSystem(RasrSystem):
    """
    Neural Network Common System Class
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
    ):
        super().__init__(rasr_binary_path=rasr_binary_path, rasr_arch=rasr_arch)

        self.returnn_root = returnn_root or (
            gs.RETURNN_ROOT if hasattr(gs, "RETURNN_ROOT") else None
        )
        self.returnn_python_home = returnn_python_home or (
            gs.RETURNN_PYTHON_HOME if hasattr(gs, "RETURNN_PYTHON_HOME") else None
        )
        self.returnn_python_exe = returnn_python_exe or (
            gs.RETURNN_PYTHON_EXE if hasattr(gs, "RETURNN_PYTHON_EXE") else None
        )

        self.blas_lib = blas_lib or (gs.BLAS_LIB if hasattr(gs, "BLAS_LIB") else None)
