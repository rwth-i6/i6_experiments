from sisyphus import *

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
def safe_csp(system, corpus='base'):
    # save initial configuration
    pristine_ur_csp = system.csp
    # temp_csp = copy.deepcopy(system.csp)
    system.csp = copy.deepcopy(system.csp)
    yield system.csp[corpus]
    # reset csp
    system.csp = pristine_ur_csp
    # system.csp = temp_csp
