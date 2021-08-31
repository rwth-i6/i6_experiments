from dataclasses import dataclass
import typing

from sisyphus import tk

from i6_core.corpus.convert import CorpusToTxtJob

from i6_experiments.users.rossenbach.sentencepiece.train import TrainSentencePiece, SentencePieceType


@dataclass(frozen=True)
class SPMSettings:
    spm_model: tk.Path
    spm_vocab_size: tk.Variable
    type: SentencePieceType


def get_spm_settings(bliss_corpus, vocab_size, model_type, **opts):
    """

    :param tk.Path bliss_corpus:
    :param int vocab_size:
    :param SentencePieceType model_type:
    :return:
    """

    text = CorpusToTxtJob(bliss_corpus, gzip=False).out_text
    train_job = TrainSentencePiece(
        training_text=text,
        vocab_size=vocab_size,
        model_type=model_type,
        **opts
        )

    settings = SPMSettings(
        spm_model=train_job.out_model,
        spm_vocab_size=train_job.out_vocab_size,
        type=model_type
    )

    return settings
