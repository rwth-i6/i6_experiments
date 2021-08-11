from dataclasses import dataclass
from functools import lru_cache
import os
import typing

from sisyphus import tk

from i6_core.bpe.train import ReturnnTrainBpeJob
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.tools.git import CloneGitRepositoryJob


@dataclass(frozen=True)
class BPESettings:
    bpe_codes: tk.Path
    bpe_vocab: tk.Path
    bpe_vocab_size: tk.Variable
    unk_label: typing.Optional[str]


@lru_cache()
def get_returnn_subword_nmt(commit_hash, output_prefix=""):
    """

    :param str commit_hash:
    :return: subword-nmt repo path
    :rtype tk.Path
    """
    subword_nmt_job = CloneGitRepositoryJob(
        url="https://github.com/albertz/subword-nmt",
        commit=commit_hash,
        checkout_folder_name="subword-nmt",
    )
    subword_nmt_job.add_alias(os.path.join(output_prefix, "clone_subword_nmt"))
    tk.register_output(os.path.join(output_prefix, "subword-nmt-repo"), subword_nmt_job.out_repository)

    return subword_nmt_job.out_repository


@lru_cache()
def get_bpe_settings(bliss_corpus, bpe_size, subword_nmt_repo_path, unk_label="UNK", output_prefix=""):
    """

    :param Path bliss_corpus
    :param int bpe_size:
    :param Path subword_nmt_repo_path:
    :param str unk_label:
    :param str output_prefix
    :return:
    :rtype: BPESettings
    """
    to_text_job = CorpusToTxtJob(bliss_corpus)
    to_text_job.add_alias(os.path.join(output_prefix, "bliss_to_text"))

    train_bpe_job = ReturnnTrainBpeJob(
        text_file=to_text_job.out_txt,
        bpe_size=bpe_size,
        unk_label=unk_label,
        subword_nmt_repo=subword_nmt_repo_path)
    train_bpe_job.add_alias(os.path.join(output_prefix, "train_bpe"))

    tk.register_output(os.path.join(output_prefix, "bpe.codes"), train_bpe_job.out_bpe_codes)
    tk.register_output(os.path.join(output_prefix, "bpe.vocab"), train_bpe_job.out_bpe_vocab)
    tk.register_output(os.path.join(output_prefix, "bpe.vocab.size"), train_bpe_job.out_vocab_size)

    return BPESettings(train_bpe_job.out_bpe_codes, train_bpe_job.out_bpe_vocab, train_bpe_job.out_vocab_size, unk_label)
