from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.serialization import Collection, PartialImport
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from ...data.base import DataConfig
from .corpus import ScorableCorpus
from .recog import AlignFunction, EncoderModel, RecogResult, SearchFunction, base_recog_forward_step, recog_base


@lru_cache(maxsize=1)
def _get_rasr_search_function(config_file: tk.Path) -> SearchFunction:
    from librasr import Configuration, SearchAlgorithm

    config = Configuration()
    config.set_from_file(config_file)

    search_algorithm = SearchAlgorithm(config=config)

    def wrapper(features: torch.Tensor) -> Tuple[str, float]:
        nonlocal search_algorithm
        traceback = search_algorithm.recognize_segment(features)
        recog_str = " ".join([traceback_item.lemma for traceback_item in traceback])
        recog_score = traceback[-1].am_score + traceback[-1].lm_score
        return recog_str, recog_score

    return wrapper


@lru_cache(maxsize=1)
def _get_rasr_align_function(config_file: tk.Path) -> AlignFunction:
    from librasr import Aligner, Configuration

    config = Configuration()
    config.set_from_file(config_file)

    aligner = Aligner(config=config)

    def wrapper(features: torch.Tensor, orth: str) -> Tuple[str, float]:
        nonlocal aligner
        traceback = aligner.align_segment(features, orth + " ")  # RASR requires a trailing space in transcription
        align_str = " ".join([traceback_item.lemma for traceback_item in traceback])
        if len(traceback) > 0:
            align_score = traceback[-1].am_score + traceback[-1].lm_score
        else:
            align_score = float("inf")
        return align_str, align_score

    return wrapper


def _rasr_recog_forward_step(
    *,
    model: EncoderModel,
    extern_data: TensorDict,
    config_file: tk.Path,
    sample_rate: int = 16000,
    align_config_file: Optional[tk.Path] = None,
    **_,
) -> None:
    base_recog_forward_step(
        model=model,
        extern_data=extern_data,
        search_function=_get_rasr_search_function(config_file),
        align_function=_get_rasr_align_function(align_config_file) if align_config_file else None,
        sample_rate=sample_rate,
    )


def recog_rasr(
    descriptor: str,
    checkpoint: PtCheckpoint,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    model_serializers: Collection,
    rasr_config_file: tk.Path,
    rasr_align_config_file: Optional[tk.Path] = None,
    sample_rate: int = 16000,
    device: Literal["cpu", "gpu"] = "cpu",
) -> RecogResult:
    rasr_forward_step_import = PartialImport(
        code_object_path=f"{_rasr_recog_forward_step.__module__}.{_rasr_recog_forward_step.__name__}",
        unhashed_package_root="",
        hashed_arguments={
            "config_file": rasr_config_file,
            "align_config_file": rasr_align_config_file,
            "sample_rate": sample_rate,
        },
        unhashed_arguments={},
        import_as="forward_step",
    )
    return recog_base(
        descriptor=descriptor,
        recog_data_config=recog_data_config,
        recog_corpus=recog_corpus,
        model_serializers=model_serializers,
        forward_step_import=rasr_forward_step_import,
        device=device,
        checkpoint=checkpoint,
        extra_output_files=["rasr.recog.log", "rasr.align.log"],
    )
