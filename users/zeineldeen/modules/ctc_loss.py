from typing import Union, List, Tuple

import returnn_common.models.layers as layers
from returnn_common.models.base import LayerRef, LayerDictRaw

from returnn.util.basic import NotSpecified


class SoftmaxCtcLossLayer(layers.Copy):
    """

    """

    def __init__(self,
                 loss_scale=1.0,
                 **kwargs):
        """
        :param str|None data_key:
        """
        super().__init__(**kwargs)
        self.loss_scale = loss_scale

    def get_opts(self):
        """
        Return all options
        """
        opts = {
            'loss_scale': self.loss_scale
        }
        opts = {key: value for (key, value) in opts.items() if value is not NotSpecified}
        return {**opts, **super().get_opts()}

    def make_layer_dict(self, source: Union[LayerRef, List[LayerRef], Tuple[LayerRef]],
                        target: LayerRef) -> LayerDictRaw:
        """
        Make layer dict
        """
        return {
            'class': 'softmax',
            'from': source,
            'loss': 'ctc',
            "loss_opts": {"beam_width": 1, "ctc_opts": {"ignore_longer_outputs_than_inputs": True}},
            'target': target,
            **self.get_opts()}
