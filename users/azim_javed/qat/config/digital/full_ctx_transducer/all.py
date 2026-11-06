from .qat_encoder import all as qat_encoder
from .qat_encoder_pred import all as qat_encoder_pred
from .qat_full_model import all as qat_full_model


def py():
    qat_encoder.py()
    qat_encoder_pred.py()
    qat_full_model.py()
