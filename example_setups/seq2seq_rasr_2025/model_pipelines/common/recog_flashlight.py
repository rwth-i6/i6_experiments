__all__ = ["recog_flashlight"]

from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.serialization import Collection, PartialImport
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from ...data.base import DataConfig
from .corpus import ScorableCorpus
from .recog import EncoderModel, RecogResult, SearchFunction, base_recog_forward_step, recog_base


@lru_cache(maxsize=1)
def _get_flashlight_search_function(
    *,
    vocab_file: tk.Path,
    lm_weight: float = 0.0,
    blank_token: str = "<blank>",
    silence_token: str = "<blank>",
    unk_word: str = "<unk>",
    **kwargs,
) -> SearchFunction:
    from torchaudio.models.decoder import ctc_decoder

    vocab = Vocabulary.create_vocab(vocab_file=vocab_file, unknown_label=None)
    assert vocab._vocab is not None
    labels = list({value: key for key, value in vocab._vocab.items()}.values())

    if "<blank>" not in labels:
        labels.append("<blank>")

    print(f"labels: {labels}")

    decoder = ctc_decoder(
        tokens=labels,
        lm_weight=lm_weight,
        blank_token=blank_token,
        silence_token=silence_token,
        unk_word=unk_word,
        **kwargs,
    )

    def wrapper(features: torch.Tensor) -> Tuple[str, float]:
        nonlocal labels
        nonlocal decoder

        hyps = decoder(-features)
        recog_str = " ".join([labels[token] for token in hyps[0][0].tokens])
        return recog_str, hyps[0][0].score

    return wrapper


def _flashlight_recog_forward_step(
    *,
    model: EncoderModel,
    extern_data: TensorDict,
    sample_rate: int = 16000,
    **kwargs,
):
    search_function = _get_flashlight_search_function(**kwargs)
    return base_recog_forward_step(
        model=model, extern_data=extern_data, search_function=search_function, sample_rate=sample_rate
    )


def recog_flashlight(
    descriptor: str,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    model_serializers: Collection,
    sample_rate: int = 16000,
    device: Literal["cpu", "gpu"] = "cpu",
    checkpoint: Optional[PtCheckpoint] = None,
    **kwargs,
) -> RecogResult:
    flashlight_forward_step_import = PartialImport(
        code_object_path=f"{_flashlight_recog_forward_step.__module__}.{_flashlight_recog_forward_step.__name__}",
        unhashed_package_root="",
        hashed_arguments={**kwargs, "sample_rate": sample_rate},
        unhashed_arguments={},
        import_as="forward_step",
    )
    return recog_base(
        descriptor=descriptor,
        recog_data_config=recog_data_config,
        recog_corpus=recog_corpus,
        model_serializers=model_serializers,
        forward_step_import=flashlight_forward_step_import,
        device=device,
        checkpoint=checkpoint,
    )
