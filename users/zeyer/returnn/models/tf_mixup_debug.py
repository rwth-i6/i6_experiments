"""
Debug code for mixup.
"""

from __future__ import annotations
from returnn.config import Config, global_config_ctx
from returnn.tf.engine import Engine
from returnn.datasets import init_dataset
from returnn.log import log
from .tf_mixup import make_mixup_layer_dict, MixupOpts


def test_mixup():
    log.initialize(verbosity=[4])
    config = Config(
        {
            "train": {
                "class": "Task12AXDataset",
                "num_seqs": 10000,
            },
            "dev": {
                "class": "Task12AXDataset",
                "num_seqs": 1000,
            },
            "extern_data": {"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            "network": {
                "mixup": make_mixup_layer_dict("data", dim=9, opts=MixupOpts()),
                # "mixup": {"class": "copy", "from": "data"},
                "lstm": {"class": "rec", "unit": "lstm", "from": "mixup", "n_out": 100},
                "output": {
                    "class": "softmax",
                    "from": "lstm",
                    "loss": "ce",
                    "target": "classes",
                },
            },
            "optimizer": {"class": "Adam"},
            "batch_size": 1000,
            "num_epochs": 5,
            "device": "cpu",
            "tf_log_memory_usage": True,
            "log_batch_size": True,
        }
    )
    with global_config_ctx(config):
        train_dataset = init_dataset(config.typed_value("train"))
        dev_dataset = init_dataset(config.typed_value("dev"))
        engine = Engine(config=config)
        engine.init_train_from_config(config=config, train_data=train_dataset, dev_data=dev_dataset)
        engine.train()


def test_profile():
    from .tf_mixup import _get_raw_func
    import tensorflow as tf
    import time

    tf.compat.v1.enable_v2_behavior()
    tf.config.set_visible_devices([], "GPU")  # cpu-only

    dim = 9
    opts = MixupOpts()
    func = _get_raw_func(dim=dim, opts=opts)

    buffer = tf.Variable(tf.zeros([opts.buffer_size, dim], dtype=tf.float32))
    buffer_pos = tf.Variable(tf.zeros((), dtype=tf.int32))
    buffer_filled = tf.Variable(tf.zeros((), dtype=tf.bool))
    train_flag = tf.constant(True)

    for n in range(1000):
        n_batch = 10
        n_time = 100
        src = tf.random.uniform([n_batch, n_time, dim], dtype=tf.float32)
        src_seq_lens = tf.random.uniform([n_batch], minval=1, maxval=n_time, dtype=tf.int32)
        start = time.time()
        func(
            src,
            src_seq_lens,
            buffer,
            buffer_pos,
            buffer_filled,
            train_flag,
        )
        print("time:", time.time() - start)
