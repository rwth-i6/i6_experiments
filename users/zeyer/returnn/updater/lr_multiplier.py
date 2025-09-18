"""
Custom LR multipliers per param name expression
"""

import torch
from collections import defaultdict
from fnmatch import fnmatchcase
from typing import Dict, List, Any
from returnn.util.basic import DictRefKeys, FrozenDict
from returnn.torch.updater import wrap_user_blacklist_wd_modules
from returnn.torch.frontend.bridge import wrapped_pt_module_to_rf_module


def optimizer_param_groups_custom_lr_multiplier(*, model: torch.nn.Module, optimizer_opts: Dict[str, Any], **_kwargs):
    """
    Use this inside the optimizer options as ``param_groups_custom`` in your RETURNN config.
    Also define ``learning_rate_multipliers_by_patterns`` in the optimizer options.

    We handle ``weight_decay_modules_blacklist`` here consistent to how RETURNN handles it.
    """
    default_weight_decay = optimizer_opts.get("weight_decay", 0.0)

    blacklist_wd_modules = wrap_user_blacklist_wd_modules(optimizer_opts.pop("weight_decay_modules_blacklist", None))
    lr_multipliers_by_patterns = optimizer_opts.pop("learning_rate_multipliers_by_patterns")

    # Tracker of visited parameters to only add each parameter once, in case two modules share common parameters.
    # We need the wrapper class RefIdEq because Parameters are compared by value and not by reference.
    params_by_opts: defaultdict[FrozenDict, List[torch.nn.Parameter]] = defaultdict(list)
    visited_params = DictRefKeys()
    for module_name, module in model.named_modules():
        module_name: str
        module: torch.nn.Module
        rf_module = wrapped_pt_module_to_rf_module(module)
        for param_name, param in module.named_parameters(recurse=False):
            param_name: str
            param: torch.nn.Parameter
            if param in visited_params:
                continue
            visited_params[param] = True
            full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name

            opts = {}
            if (
                param_name.endswith("bias")
                or isinstance(module, blacklist_wd_modules)
                or isinstance(rf_module, blacklist_wd_modules)
            ):
                opts["weight_decay"] = 0.0
            else:
                opts["weight_decay"] = default_weight_decay
            # Should not depend on order in lr_multipliers_by_patterns.
            matching = {
                pattern: lr_multiplier
                for pattern, lr_multiplier in lr_multipliers_by_patterns.items()
                if fnmatchcase(full_param_name, pattern)
            }
            assert len(matching) <= 1, f"expect no match or exactly one match, got matches: {matching}"
            if matching:
                lr_multiplier = next(iter(matching.values()))
                if lr_multiplier != 1.0:
                    opts["learning_rate_multiplier"] = lr_multiplier
            params_by_opts[FrozenDict(opts)].append(param)

    return [{"params": params, **opts} for opts, params in params_by_opts.items()]
