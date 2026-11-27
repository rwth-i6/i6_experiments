from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import (
    get_returnn_subword_nmt,
    get_bpe_settings,
    BPESettings,
)
from .corpus import get_bliss_corpus_dict


def get_subword_nmt_bpe(bpe_size: int, unk_label: str = "<unk>", subdir_prefix: str = "") -> BPESettings:
    """
    Get the BPE tokens via the Returnn subword-nmt for a Tedlium2 setup.

    :param bpe_size: the number of BPE merge operations. This is NOT the resulting vocab size!
    :param unk_label: unknown label symbol
    :param subdir_prefix: dir name prefix for aliases and outputs
    """
    subword_nmt_repo = get_returnn_subword_nmt(output_prefix=subdir_prefix)
    train_corpus = get_bliss_corpus_dict()["train"]
    bpe_settings = get_bpe_settings(
        train_corpus,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=subdir_prefix,
        subword_nmt_repo_path=subword_nmt_repo,
    )
    return bpe_settings


def get_subword_nmt_bpe_v2(bpe_size: int, unk_label: str = "<unk>", subdir_prefix: str = "") -> BPESettings:
    """
    Get the BPE tokens via the Returnn subword-nmt for a Tedlium2 setup.

    V2: Uses subword-nmt version corrected for Apptainer related bug, adds hash overwrite for repo

    :param bpe_size: the number of BPE merge operations. This is NOT the resulting vocab size!
    :param unk_label: unknown label symbol
    :param subdir_prefix: dir name prefix for aliases and outputs
    """
    subword_nmt_repo = get_returnn_subword_nmt(
        commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf", output_prefix=subdir_prefix
    )
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"
    train_corpus = get_bliss_corpus_dict()["train"]
    bpe_settings = get_bpe_settings(
        train_corpus,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=subdir_prefix,
        subword_nmt_repo_path=subword_nmt_repo,
    )
    return bpe_settings
