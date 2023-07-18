from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import (
    get_returnn_subword_nmt,
    get_bpe_settings,
    BPESettings,
)


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
