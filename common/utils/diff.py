
"""
Diff utils
"""


from sisyphus import gs, tk

import os
from typing import List

import i6_core.rasr as rasr
import i6_core.util

import i6_experiments.common.setups.rasr.util as rasr_util
from .repr import py_repr


_limit = 3


def collect_diffs(prefix: str, orig, new) -> List[str]:
    """
    :return: list of diff descriptions. empty if no diffs
    """
    if orig is None and new is None:
        return []
    if isinstance(orig, i6_core.util.MultiPath) and isinstance(new, i6_core.util.MultiPath):
        pass  # allow different sub types
    elif type(orig) != type(new):
        return [f"{prefix} diff type: {py_repr(orig)} != {py_repr(new)}"]
    if isinstance(orig, dict):
        diffs = collect_diffs(f"{prefix}:keys", set(orig.keys()), set(new.keys()))
        if diffs:
            return diffs
        num_int_key_diffs = 0
        keys = list(orig.keys())
        for i in range(len(keys)):
            key = keys[i]
            sub_diffs = collect_diffs(f"{prefix}[{key!r}]", orig[key], new[key])
            diffs += sub_diffs
            if isinstance(key, int) and sub_diffs:
                num_int_key_diffs += 1
            if num_int_key_diffs >= _limit and i < len(keys) - 1:
                diffs += [f"{prefix} ... ({len(keys) - i - 1} remaining)"]
                break
        return diffs
    if isinstance(orig, set):
        sorted_orig = sorted(orig)
        sorted_new = sorted(new)
        i, j = 0, 0
        num_diffs = 0
        diffs = []
        while i < len(sorted_orig) or j < len(sorted_new):
            if i < len(sorted_orig) and j < len(sorted_new):
                if sorted_orig[i] < sorted_new[j]:
                    cmp = -1
                elif sorted_orig[i] > sorted_new[j]:
                    cmp = 1
                else:
                    cmp = 0
            elif i >= len(sorted_orig):
                cmp = 1
            elif j >= len(sorted_new):
                cmp = -1
            else:
                assert False
            if cmp != 0:
                num_diffs += 1
                if num_diffs <= _limit:
                    diffs += [
                        f"{prefix} diff: del {py_repr(sorted_orig[i])}"
                        if cmp < 0 else
                        f"{prefix} diff: add {py_repr(sorted_new[j])}"
                    ]
                if cmp < 0:
                    i += 1
                else:
                    j += 1
            else:
                i += 1
                j += 1
        if num_diffs > _limit:
            diffs += [f"{prefix} ... ({num_diffs - _limit} remaining)"]
        return diffs
    if isinstance(orig, (list, tuple)):
        if len(orig) != len(new):
            return [f"{prefix} diff len: {py_repr(orig)} != {py_repr(new)}"]
        diffs = []
        num_diffs = 0
        for i in range(len(orig)):
            sub_diffs = collect_diffs(f"{prefix}[{i}]", orig[i], new[i])
            diffs += sub_diffs
            if sub_diffs:
                num_diffs += 1
            if num_diffs >= _limit and i < len(orig) - 1:
                diffs += [f"{prefix} ... ({len(orig) - i - 1} remaining)"]
                break
        return diffs
    if isinstance(orig, (int, float, str)):
        if orig != new:
            return [f"{prefix} diff: {py_repr(orig)} != {py_repr(new)}"]
        return []
    if isinstance(orig, tk.AbstractPath):
        return collect_diffs(f"{prefix}:path-state", _PathState(orig), _PathState(new))
    if isinstance(orig, i6_core.util.MultiPath):
        # only hidden_paths relevant (?)
        return collect_diffs(f"{prefix}.hidden_paths", orig.hidden_paths, new.hidden_paths)
    if isinstance(orig, _expected_obj_types):
        orig_attribs = set(vars(orig).keys())
        new_attribs = set(vars(new).keys())
        diffs = collect_diffs(f"{prefix}:attribs", orig_attribs, new_attribs)
        if diffs:
            return diffs
        for key in vars(orig).keys():
            diffs += collect_diffs(f"{prefix}.{key}", getattr(orig, key), getattr(new, key))
        return diffs
    raise TypeError(f"unexpected type {type(orig)}")


class _PathState:
    """
    Wraps AbtractPath in a way such that the hash behavior is the same
    """

    def __init__(self, p: tk.AbstractPath):
        # Adapted from AbstractPath._sis_hash and a bit simplified:
        assert not isinstance(p.creator, str)
        if p.hash_overwrite is None:
            creator = p.creator
            path = p.path
        else:
            overwrite = p.hash_overwrite
            assert_msg = "sis_hash for path must be str or tuple of length 2"
            if isinstance(overwrite, tuple):
                assert len(overwrite) == 2, assert_msg
                creator, path = overwrite
            else:
                assert isinstance(overwrite, str), assert_msg
                creator = None
                path = overwrite
        if hasattr(creator, '_sis_id'):
            creator = creator._sis_id()  # noqa
        elif isinstance(creator, str) and creator.endswith(f"/{gs.JOB_OUTPUT}"):
            creator = creator[:-len(gs.JOB_OUTPUT) - 1]
        if isinstance(creator, str):
            # Ignore the full name and job hash.
            creator = os.path.basename(creator)
            creator = creator.split(".")[0]
        self.creator = creator
        self.path = path


_expected_obj_types = (
    rasr_util.RasrInitArgs,
    rasr_util.ReturnnRasrDataInput,
    rasr.CommonRasrParameters,
    rasr.RasrConfig,
    rasr.FlowNetwork,
    rasr.NamedFlowAttribute,
    rasr.FlagDependentFlowAttribute,
    _PathState,
)
