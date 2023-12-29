"""
Layerdrop, the most simple way, like ESPnet

https://github.com/espnet/espnet/blob/7c140c2ac9b4f642acb36131217dd984d4601681/espnet2/asr/encoder/conformer_encoder.py#L278
https://github.com/espnet/espnet/blob/7c140c2ac9b4f642acb36131217dd984d4601681/espnet/nets/pytorch_backend/transformer/repeat.py#L29
"""

from __future__ import annotations
from typing import Optional, Dict
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


class SequentialLayerDrop(rf.Sequential):
    def __init__(self, *args, layer_drop: float):
        super().__init__(*args)
        self.layer_drop = layer_drop

    def __call__(self, inp, *, collected_outputs: Optional[Dict[str, Tensor]] = None, **kwargs) -> Tensor:
        def _layer_drop_call():
            x = inp
            num_layers_dim = Dim(len(self), name="num_layers")
            drop_probs = rf.random_uniform([num_layers_dim])
            for i, (name, module) in enumerate(self.items()):
                x = rf.cond(
                    drop_probs[i] >= self.layer_drop,
                    lambda: module(x, **kwargs),
                    lambda: x,
                )
                if collected_outputs is not None:
                    collected_outputs[name] = x
            return x

        def _no_layer_drop_call():
            return super()(inp, collected_outputs=collected_outputs, **kwargs)

        return rf.cond(rf.get_run_ctx().train_flag, _layer_drop_call, _no_layer_drop_call)
