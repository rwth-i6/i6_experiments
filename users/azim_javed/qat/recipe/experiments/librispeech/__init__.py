from typing import Dict, List, Optional, Tuple

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from ...model_pipelines.common.train import TrainedModel
from . import recognition, training


def run_all(filename):
    baseline_qat_config = dict(
        weight_bit_prec=8,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    models = {
        "qat_ffnn_transducer_full_quant": training.qat_ffnn_transducer_bpe.run(descriptor="qat_ffnn_transducer_full_quant", qat_args=baseline_qat_config),
        "ffnn_transducer_qat_encoder": training.ffnn_transducer_qat_encoder_bpe.run(descriptor="ffnn_transducer_qat_encoder", qat_args=baseline_qat_config),
        "qat_ctc_bpe": training.qat_ctc_bpe.run(descriptor="qat_ctc_bpe", qat_args=baseline_qat_config),
    }
    recog_results = []
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer"]))
    # register_recog_report(recog_results, filename=filename)
    return models, recog_results

def run_debug(filename):
    baseline_qat_config = dict(
        weight_bit_prec=8,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    models = {
        "qat_ctc_bpe": training.qat_ctc_bpe.run(descriptor="qat_ctc_bpe", qat_args=baseline_qat_config),

    }
    recog_results = []
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer"]))
    # register_recog_report(recog_results, filename=filename)
    return models, recog_results