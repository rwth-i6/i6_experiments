"""
Serialization utils for returnn torch
"""
from typing import Optional

from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection, Call,
    build_config_constructor_serializers,
)
from i6_core.returnn.config import CodeWrapper

def returnn_torch_init_model(
    cfg: ModelConfiguration,
    model_variable: str,
):
    """
    This is to load a model directly to returnn torch config.
    Something like

    ```
    import all_needed_stuffs_here
    lm_cfg = LSTMLMConfig(...)
    lm = LSTMLM(cfg=lm_cfg)
    ```

    and then `lm` can be passed to the train step.

    :param cfg: Model Config
    :param model_variable: name to assign to the model
    :returns: Collection object to do all of these
    """
    cfg_call, model_imports = build_config_constructor_serializers(cfg, f"{model_variable}_cfg")
    model_call = Call(
        cfg.__class__.__name__,
        kwargs=[("epoch", 0), ("step", 0), ("cfg", f"{model_variable}_cfg")]
    )
    
    serializer_objects = [
        model_imports,
        cfg_call,
        model_call,
    ]
    return Collection(serializer_objects=serializer_objects)


def returnn_torch_load_ckpt(
    cfg: ModelConfiguration,
    model_variable: str,
    model_checkpoint_path: Optional[str] = None,
    model_checkpoint_variable: Optional[str] = None,
):
    """
    Load a model directly in returnn torch config

    ```
    lm = <Construct nn.Module here>
    lm_ckpt = torch.load("path/to/checkpoint")
    lm.load_state_dict(lm_ckpt["model"])
    ```

    :param cfg: Model Config
    :param model_variable: name to assign to the model
    :param model_checkpoint_path: Path to returnn torch checkpoint if prelaod
    :param model_checkpoint_variable: name to assign to the torch.load(checkpoint)
    :returns: Collection object to do all of these
    """
    load_ckpt_call = Call(
        "torch.load",
        kwargs=[("f", model_checkpoint_path)],
        return_assign_variables=model_checkpoint_variable
    )
    load_state_dict_call = Call(
        f"{model_variable}.load_state_dict",
        kwargs=[("state_dict", f"{model_checkpoint_variable}['model']")]
    )
    serializer_objects = [
        load_ckpt_call,
        load_state_dict_call,
    ]
    return Collection(serializer_objects=serializer_objects)
