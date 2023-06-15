from .encoder import get_feature_extraction_dict, get_conformer_dict
from .decoder import get_decoder_dict


def get_net_dict():
  net_dict = {}

  net_dict.update(get_feature_extraction_dict())
  net_dict.update(get_conformer_dict())
  net_dict.update(get_decoder_dict())

  return net_dict
