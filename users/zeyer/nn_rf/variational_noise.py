"""
Variational noise (weight noise) helpers

(Variational/weight noise itself is already in RETURNN, rf.weight_noise)
"""

from typing import Optional, Union, Any, Dict, Tuple
from fnmatch import fnmatchcase
from returnn.config import Config, get_global_config
import returnn.frontend as rf
from returnn.util.basic import DictRefKeys


def maybe_apply_variational_noise_from_config(model: rf.Module, config: Optional[Config] = None):
    """maybe apply variational noise from config (if set)"""
    if not config:
        config = get_global_config()

    vn = config.typed_value("variational_noise", None)
    if vn:
        # Old style. Warning: Not recommended like this.
        # Better/different defaults with new-style below. (E.g. min_ndim>=2.)
        # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
        # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
        blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
        blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
        for module_name, module in model.named_modules():
            if isinstance(module, blacklist):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                if param_name.endswith("bias"):  # no bias
                    continue
                if param.auxiliary:
                    continue
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name
                print(f"{full_param_name}: applying variational noise with std={vn}")
                rf.weight_noise(module, param_name, std=vn)

    # See also i6_experiments.users.zeyer.returnn.updater.lr_multiplier.optimizer_param_groups_custom_lr_multiplier
    # as another example of how to use such patterns.
    vn_by_pattern: Optional[Dict[str, Union[float, Dict[str, Any]]]] = config.typed_value(
        "variational_noise_by_pattern", None
    )
    if vn_by_pattern:
        assert isinstance(vn_by_pattern, dict)
        vn_by_pattern = {pattern: _transform_opts(opts) for pattern, opts in vn_by_pattern.items()}
        blacklists = {k: _get_module_blacklist_from_opts(opts, config) for k, opts in vn_by_pattern.items()}
        blacklists_set = set(blacklists.values())
        visited_params = DictRefKeys()
        for module_name, module in model.named_modules():
            module_name: str
            module: rf.Module
            if all(isinstance(module, blacklist) for blacklist in blacklists_set):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                param_name: str
                param: rf.Parameter
                if param in visited_params:
                    continue
                visited_params[param] = True
                if param.auxiliary:
                    continue
                if not param.dtype.startswith("float") and not param.dtype.startswith("bfloat"):
                    continue
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name

                # Should not depend on order in variational_noise_by_pattern.
                matching = {
                    pattern: opts
                    for pattern, opts in vn_by_pattern.items()
                    if fnmatchcase(full_param_name, pattern)
                    and not isinstance(module, blacklists[pattern])
                    and param.ndim >= opts.get("min_ndim", 2)
                    and not any(
                        fnmatchcase(full_param_name, ex_pattern)
                        for ex_pattern in opts.get("exclude_patterns", ["*bias"])
                    )
                }
                assert len(matching) <= 1, f"expect no match or exactly one match, got matches: {matching}"
                if matching:
                    opts = next(iter(matching.values()))
                    opts = opts.copy()  # copy to not modify the original config
                    opts.pop("modules_blacklist", None)  # not needed here
                    opts.pop("min_ndim", None)
                    opts.pop("exclude_patterns", None)
                    std = opts.pop("std")
                    assert not opts, f"unexpected opts: {opts}"
                    print(f"{full_param_name}: applying variational noise with std={vn}")
                    rf.weight_noise(module, param_name, std=std)


def _transform_opts(v: Union[float, Dict[str, Any]]) -> Dict[str, Any]:
    """Transform the options for variational noise."""
    if isinstance(v, float):
        return {"std": v}
    elif isinstance(v, dict):
        return v
    else:
        raise TypeError(f"Unexpected type for variational noise options: {type(v)}")


def _get_module_blacklist_from_opts(opts: Dict[str, Any], config: Config) -> Tuple[type, ...]:
    """Get the blacklist from the opts."""
    if "modules_blacklist" in opts:
        blacklist = opts["modules_blacklist"]
    else:
        blacklist = config.typed_value("optimizer").get("weight_decay_modules_blacklist")
    if blacklist is None:
        return ()
    blacklist = tuple(eval(entry, {"rf": rf}) if isinstance(entry, str) else entry for entry in blacklist)
    assert all(issubclass(cls, rf.Module) for cls in blacklist), (
        f"blacklist must contain rf.Module subclasses, got {blacklist}"
    )
    return blacklist
