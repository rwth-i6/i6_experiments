from i6_core.returnn.config import ReturnnConfig

from .helper import get_network
from .helper import make_nn_config



def get_wei_config():
  # specaug codes not included
  # train and dev dataset not included
  # partition_epochs: 5 for training data
  network = get_network()
  nn_config = make_nn_config(network)
  nn_config['extern_data'] = {
    'data': {"dim": 80, "shape": (None, 80), "available_for_inference": True},  # input: 80-dimensional logmel features
    'classes': {"dim": 9001, "shape": (None,), "available_for_inference": True, "sparse": True, "dtype": "int8"}
  }



  # train, dev dataset not defined
  # returnn_config = ReturnnConfig(config=nn_config,
  #                                python_prolog=None, # add python codes for specaug?
  #                                )

  return nn_config