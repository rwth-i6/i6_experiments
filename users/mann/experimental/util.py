from sisyphus import *

from i6_core.returnn import ReturnnConfig

import contextlib
import copy

@contextlib.contextmanager
def new_rasr():
    temp_sprint_root = gs.SPRINT_ROOT
    temp_tf_native_ops = gs.TF_NATIVE_OPS
    gs.SPRINT_ROOT = '/work/tools/asr/rasr/20191102_generic/'
    gs.TF_NATIVE_OPS = '/work/tools/asr/returnn_native_ops/20190919_0e23bcd20/generic/NativeLstm2/NativeLstm2.so'
    try:
        yield
    finally:
        gs.SPRINT_ROOT = temp_sprint_root
        gs.TF_NATIVE_OPS = temp_tf_native_ops

@contextlib.contextmanager
def safe_crp(system, corpus='base'):
    # save initial configuration
    pristine_ur_crp = system.crp
    # temp_csp = copy.deepcopy(system.csp)
    system.crp = copy.deepcopy(system.crp)
    yield system.crp[corpus]
    # reset csp
    system.crp = pristine_ur_crp
    # system.csp = temp_csp


def _RelaxedOverwriteConfig(ReturnnConfig):

    def check_consistency():
        """Intercept overwriting error."""
        try:
            super().check_consistency()
        except AssertionError as ae:
            if "post_config would overwrite existing entry in config" not in str(ae):
                raise ae
