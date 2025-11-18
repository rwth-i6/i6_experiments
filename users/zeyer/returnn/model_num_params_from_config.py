from typing import Optional
from sisyphus import Job, Task, tk
from i6_core.returnn.config import ReturnnConfig


class GetNumParamsFromReturnnConfigJob(Job):
    """
    Get num params
    """

    def __init__(
        self,
        returnn_config: ReturnnConfig,
        *,
        returnn_root: Optional[tk.Path] = None,
    ):
        self.returnn_config = returnn_config
        self.returnn_root = returnn_root

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_num_params = self.output_var("num_params.txt")

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", mini_task=True)

    def create_files(self):
        self.returnn_config.write(self.out_returnn_config_file.get_path())

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.log import log
        from returnn.util.basic import describe_returnn_version, BehaviorVersion
        from returnn.util import better_exchook
        from returnn.config import Config
        from returnn.util import basic as util
        from returnn.config import global_config_ctx
        import returnn.frontend as rf
        from returnn.torch.frontend.bridge import rf_module_to_pt_module
        import torch

        print(f"{self}, RETURNN {describe_returnn_version()}")

        log.initialize(verbosity=[5])
        better_exchook.install()

        config = Config()
        config.load_file(self.out_returnn_config_file.get_path())

        BehaviorVersion.set(config.int("behavior_version", None))

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
                raise TypeError(
                    f"get_model returned {model} of type {type(model)}, expected rf.Module or torch.nn.Module"
                )
            assert isinstance(pt_model, torch.nn.Module)
            print("Model:", pt_model)
            num_params = sum([parameter.numel() for parameter in pt_model.parameters()])
            print(f"net params #: {num_params}")
            self.out_num_params.set(num_params)
