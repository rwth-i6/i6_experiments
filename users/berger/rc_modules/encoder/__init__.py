from enum import Enum, auto
from typing import Union

from returnn_common.nn.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder
from .conformer import VGGConformer
from .blstm import Blstm


class EncoderType(Enum):
    Blstm = auto()
    Conformer = auto()


def make_encoder(type: EncoderType, *args, **kwargs) -> Union[ISeqFramewiseEncoder, ISeqDownsamplingEncoder]:
    return {EncoderType.Blstm: Blstm, EncoderType.Conformer: VGGConformer}[type](*args, **kwargs)
