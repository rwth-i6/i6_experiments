__all__ = ["run"]

from typing import List, Optional

from ....data.librispeech import datasets as librispeech_datasets
from ....model_pipelines.aed.pytorch_modules import AEDI6DecoderConfig, AEDI6DecoderEncoder
from ....model_pipelines.common.recog import RecogResult
from ....model_pipelines.common.train import TrainedModel
from . import aed_bpe
from .aed_bpe import AEDRecogVariant


def run(
    model: TrainedModel[AEDI6DecoderConfig],
    variants: Optional[List[AEDRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    return aed_bpe.run(
        model=model,
        variants=variants,
        corpora=corpora,
        encoder_class=AEDI6DecoderEncoder,
    )
