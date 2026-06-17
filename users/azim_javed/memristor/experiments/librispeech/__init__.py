from typing import Dict, List, Optional, Tuple

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from ...model_pipelines.common.train import TrainedModel
from . import recognition, training


def run_all(filename):
    models = {
        "qat_ffnn_transducer": training.qat_ffnn_transducer_bpe.run(descriptor="qat_ffnn_transducer"),
    }
    recog_results = []
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer"]))
    # register_recog_report(recog_results, filename=filename)
    return models, recog_results