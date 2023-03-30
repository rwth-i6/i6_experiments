"""
Test for serialization
"""

from __future__ import annotations
import pickle
from returnn.tf.util.data import batch_dim, single_step_dim, SpatialDim, FeatureDim
from .serialization import *


def test_get_serializable_config_dims():
    import pprint

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("out", 5)
    config = ReturnnConfig(
        {
            "network": {
                "output": {
                    "class": "linear",
                    "out_dim": feat_dim,
                    "out_shape": {batch_dim, time_dim, feat_dim},
                },
            }
        }
    )
    config = get_serializable_config(config)
    print(config._ReturnnConfig__parse_python(config.python_prolog))
    print(config.config)
    print(config._ReturnnConfig__parse_python(config.python_epilog))
    for k, v in config.config.items():
        assert pprint.isreadable(v)
    assert config.python_prolog


def test_get_serializable_config_dims_single_step():
    import pprint

    feat_dim = FeatureDim("out", 5)
    config = ReturnnConfig(
        {
            "network": {
                "output": {
                    "class": "rec",
                    "axis": single_step_dim,
                    "out_dim": feat_dim,
                },
            }
        }
    )
    config = get_serializable_config(config)
    config_dump = pickle.dumps(config)
    config = pickle.loads(config_dump)
    print(config._ReturnnConfig__parse_python(config.python_prolog))
    print(config.config)
    print(config._ReturnnConfig__parse_python(config.python_epilog))
    for k, v in config.config.items():
        assert pprint.isreadable(v)
    assert config.python_prolog


def test_get_serializable_config_function():
    import pprint

    config = ReturnnConfig(
        {
            "network": {
                "output": {
                    "class": "eval",
                    "eval": _test_get_serializable_config_function_test_func,
                },
            }
        }
    )
    config = get_serializable_config(config)
    print(config._ReturnnConfig__parse_python(config.python_prolog))
    print(config.config)
    print(config._ReturnnConfig__parse_python(config.python_epilog))
    for k, v in config.config.items():
        assert pprint.isreadable(v)


def _test_get_serializable_config_function_test_func(source, **_kwargs):
    """for testing the function serialization"""
    return source(0)
