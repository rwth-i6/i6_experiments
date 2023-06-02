"""
Debug code for mixup.
"""

from __future__ import annotations
from returnn.config import Config, global_config_ctx
from returnn.tf.engine import Engine
from returnn.datasets import init_dataset
from .tf_mixup import make_mixup_layer_dict, MixupOpts


def test_mixup():
    config = Config(
        {
            "train": {
                "class": "Task12AXDataset",
                "num_seqs": 1000,
            },
            "extern_data": {"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            "network": {
                "mixup": make_mixup_layer_dict("data", dim=9, opts=MixupOpts()),
                "output": {"class": "softmax", "from": "mixup", "loss": "ce", "target": "classes"},
            },
            "num_epochs": 5,
        }
    )
    with global_config_ctx(config):
        train_dataset = init_dataset(config.typed_value("train"))
        engine = Engine(config=config)
        engine.init_train_from_config(config=config, train_data=train_dataset, dev_data=None)
        engine.train()
