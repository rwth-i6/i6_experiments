"""
Take some Sisyphus config,
collect training jobs,
check the RETURNN config,
build model
(but not on a real device, using some dummy device, e.g. Torch Meta device, to not consume memory),
and count the number of parameters dynamically.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import os
import sys
import logging
import argparse
import time
from functools import reduce

if TYPE_CHECKING:
    from returnn.config import Config
    from i6_core.returnn.training import ReturnnTrainingJob


# It will take the dir of the checked out git repo.
# So you can also only use it there...
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = f"{_setup_base_dir}/tools/sisyphus"


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)

        os.environ["SIS_GLOBAL_SETTINGS_FILE"] = f"{_setup_base_dir}/settings.py"

        try:
            import sisyphus  # noqa
            import i6_experiments  # noqa
        except ImportError:
            print("setup base dir:", _setup_base_dir)
            print("sys.path:")
            for path in sys.path:
                print(f"  {path}")
            raise


_setup()


def main():
    from sisyphus import tk
    from sisyphus.loader import config_manager
    from i6_core.returnn.training import ReturnnTrainingJob

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("config_files", nargs="*")
    args = arg_parser.parse_args()

    start = time.time()
    config_manager.load_configs(args.config_files)
    load_time = time.time() - start
    logging.info("Config loaded (time needed: %.2f)" % load_time)

    for job in tk.sis_graph.jobs():
        if not isinstance(job, ReturnnTrainingJob):
            continue
        if not job.get_aliases():
            print("Skipping training job without alias:", job)
            continue
        print(f"Training job: {job}")
        cfg = dynamic_returnn_config_from_returnn_training_job(job)
        num_params = get_num_params_from_returnn_config(cfg)
        print(f"  Number of parameters: {num_params}")


def dynamic_returnn_config_from_returnn_training_job(job: ReturnnTrainingJob) -> Config:
    """
    :param job:
    :return: RETURNN config
    """
    from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy
    from i6_core.returnn.config import ReturnnConfig
    from returnn.config import Config

    # See i6_core.returnn.training.ReturnnTrainingJob.create_files
    # See i6_core.returnn.training.ReturnnTrainingJob.create_returnn_config
    # See i6_core.returnn.config.ReturnnConfig.write
    # See i6_core.returnn.config.ReturnnConfig._serialize

    # Note, we don't need to do all of that.
    # We try to do just enough here that it works.

    job_cfg: ReturnnConfig = job.returnn_config
    cfg_dict = instanciate_delayed_copy(job_cfg.config)
    return Config(cfg_dict)


def get_num_params_from_returnn_config(config: Config) -> int:
    """
    :param config: WARNING: we might modify the config here
    :return: number of parameters
    """
    from returnn.util import basic as util
    from returnn.config import global_config_ctx
    import returnn.frontend as rf
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    import torch

    extern_data_dict = config.typed_value("extern_data")
    for k, v in extern_data_dict.items():
        assert isinstance(k, str) and isinstance(v, dict), f"invalid extern_data entry {k}: {v}"
        if "vocab" in v:
            assert "dim" in v or "sparse_dim" in v
            # TODO dummy vocab, _returnn_get_model wants that...

    rf.select_backend_torch()

    with rf.set_default_device_ctx("meta"), global_config_ctx(config):
        # See returnn.torch.engine.Engine._create_model
        get_model_func = config.typed_value("get_model")
        assert get_model_func, "get_model not defined in config"
        sentinel_kw = util.get_fwd_compat_kwargs()
        model = get_model_func(epoch=1, step=0, device=rf.get_default_device(), **sentinel_kw)
        if isinstance(model, rf.Module):
            pt_model = rf_module_to_pt_module(model)
        elif isinstance(model, torch.nn.Module):
            pt_model = model
        else:
            raise TypeError(f"get_model returned {model} of type {type(model)}, expected rf.Module or torch.nn.Module")
        assert isinstance(pt_model, torch.nn.Module)
        # print("Model:", pt_model)
        num_params = sum([parameter.numel() for parameter in pt_model.parameters()])
        return num_params


if __name__ == "__main__":
    main()
