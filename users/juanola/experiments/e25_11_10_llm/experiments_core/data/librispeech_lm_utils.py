import os
from functools import cache
from functools import lru_cache
from typing import Dict, Union, Any

from sisyphus import tk

from i6_core.corpus import CorpusToTextDictJob
from i6_core.text.convert import TextDictToTextLinesJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from i6_core.tools.download import DownloadJob
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel

"""
NEW LM CODE - mostly from Alberts librispeech file
"""

_alias_prefix = "datasets/LibriSpeech/"


def get_librispeech_lm_combined_txt() -> tk.Path:
    from i6_core.text.processing import ConcatenateJob
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

    return ConcatenateJob([get_librispeech_normalized_lm_data(), get_librispeech_train_corpus_text()]).out


@lru_cache
def get_librispeech_normalized_lm_data(output_prefix="datasets") -> tk.Path:  # From librispeech alberts code
    """
    Download the official normalized LM data for LibriSpeech

    :param output_prefix:
    :return: gzipped text file containing the LM training data
    """
    download_job = DownloadJob(url="https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz")
    download_job.add_alias(os.path.join(output_prefix, "LibriSpeech", "lm", "download_lm_data"))
    return download_job.out_file


@cache
def get_librispeech_train_corpus_text(key="train-other-960") -> tk.Path:  # From librispeech alberts code
    """train corpus text (used for LM training)"""
    train_corpus_text_dict = _get_corpus_text_dict(key)
    job = TextDictToTextLinesJob(train_corpus_text_dict, gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_lines.txt.gz", job.out_text_lines)
    return job.out_text_lines


@cache
def _get_corpus_text_dict(key: str) -> tk.Path:
    job = CorpusToTextDictJob(_get_bliss_corpus_dict()[key], gzip=True)
    job.add_alias(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict")
    tk.register_output(_alias_prefix + f"{key.replace('-', '_')}_corpus_text_dict.py.gz", job.out_dictionary)
    return job.out_dictionary


@cache
def _get_bliss_corpus_dict() -> Dict[str, tk.Path]:
    # Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
    # WARNING: Do not use these directly... It will keep another ogg copy of the audio...
    # However, these are used later in the scoring, so when changing them, make sure it's optional,
    # to not break hashes of old setups.
    return get_bliss_corpus_dict(audio_format="ogg")

@cache
def _get_spm_vocab(
        *,
        dim: Union[int, str],
        model_type: SentencePieceType = SentencePieceType.UNIGRAM,

        use_train_corpus_text: bool = True,
        use_normalized_lm_data: bool = False
) -> SentencePieceModel:
    dim_str = str(dim)
    if isinstance(dim, str):
        # Not sure if power-of-two or just multiple-of-64, but 10240 has more 2s in it (2048*5) than 10048.
        dim = {"20k": 20_480, "10k": 10_240, "5k": 5_120, "4k": 4_096, "1k": 1_024, "512": 512, "128": 128, "64": 64}[
            dim
        ]
    assert isinstance(dim, int) and dim >= 10

    if use_train_corpus_text and not use_normalized_lm_data:
        training_text = get_librispeech_train_corpus_text()
        large_training = False
        name_postfix = "_train_corpus"
    elif not use_train_corpus_text and use_normalized_lm_data:
        training_text = get_librispeech_normalized_lm_data()
        large_training = True
        name_postfix = "_norm_lm"
    elif use_train_corpus_text and use_normalized_lm_data:
        training_text = get_librispeech_lm_combined_txt()
        large_training = True
        name_postfix = "_full"
    else:
        raise ValueError("At least one corpus is needed for SPM training.")

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    _spm_train_job = TrainSentencePieceJob(
        training_text=training_text,
        vocab_size=dim,
        model_type=model_type,
        additional_options={
            "split_digits": True,
            "unk_id": 2,  # default is 0
            "bos_id": 1,  # default is 1
            "eos_id": 0,  # default is 2
            **(
                {
                    "train_extremely_large_corpus": True,
                    "shuffle_input_sentence": True,
                    "input_sentence_size": 10_000_000,  # oom otherwise, with full (40M), it takes more than 126GB
                }
                if large_training
                else {}
            ),
        },
    )
    if large_training:
        _spm_train_job.rqmt.update({"time": 12, "mem": 126})  # needs much more mem, maybe little longer
    _spm_train_job.add_alias(f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train{name_postfix}")
    tk.register_output(
        f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train{name_postfix}.model", _spm_train_job.out_model
    )
    tk.register_output(
        f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train{name_postfix}.vocab",
        ExtractSentencePieceVocabJob(_spm_train_job.out_model).out_vocab,
    )
    spm = SentencePieceModel(
        dim=dim,
        model_file=_spm_train_job.out_model,
        unknown_label="<unk>",
        bos_idx=1,
        eos_idx=0,
    )
    return spm


def get_librispeech_spm_datastream(vocab_size: int,
                                   use_train_corpus_text: bool = True,
                                   use_normalized_lm_data: bool = False
                                   ) -> SentencePieceDatastream:
    """
    Returns the datastream for the spm labels.
    Computes the SPM vocabulary.

    :param vocab_size: the size of the vocabulary
    """
    spm_model: SentencePieceModel = _get_spm_vocab(
        dim=vocab_size,
        use_train_corpus_text=use_train_corpus_text,
        use_normalized_lm_data=use_normalized_lm_data
    )
    return SentencePieceDatastream(
        available_for_inference=False,
        spm_model=spm_model.model_file,
        vocab_size=vocab_size,
    )

def get_extern_data_data() -> Dict[str, Dict[str, Any]]:
    """
    Get extern data
    """
    from returnn.tensor import Dim, batch_dim

    out_spatial_dim = Dim(None, name="out-spatial", kind=Dim.Types.Spatial)
    #classes_dim = Dim(self.vocab.get_num_classes(), name="vocab", kind=Dim.Types.Spatial)

    return  {
            "dim": 10240,     # vocab size
            "sparse": True,

            #"dim_tags": [batch_dim, out_spatial_dim],
            #"sparse": True,
            #"sparse_dim": classes_dim,
            #"vocab": self.vocab.get_opts(),
        }
