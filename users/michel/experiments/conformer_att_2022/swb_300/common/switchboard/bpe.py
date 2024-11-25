from i6_experiments.common.datasets.switchboard.corpus_train import (
    get_spoken_form_train_bliss_corpus_ldc,
    get_train_bliss_corpus_ldc,
)
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import (
    get_returnn_subword_nmt,
    get_bpe_settings,
    BPESettings,
)


def get_subword_nmt_bpe(
    bpe_size: int, use_spoken_form: bool = True, unk_label: str = "<unk>", subdir_prefix: str = ""
) -> BPESettings:
    """
    Get the BPE tokens via the Returnn subword-nmt for a Switchboard setup.

    WARNING: This will not give compatible BPE to old setups!

    :param bpe_size: the number of BPE merge operations. This is NOT the resulting vocab size!
    :param use_spoken_form: use the legacy conversion table to convert words into their spoken form
    :param unk_label: unknown label symbol
    :param subdir_prefix:
    """
    subword_nmt_repo = get_returnn_subword_nmt(output_prefix=subdir_prefix)
    if use_spoken_form:
        train_corpus = get_spoken_form_train_bliss_corpus_ldc()
    else:
        train_corpus = get_train_bliss_corpus_ldc()
    bpe_settings = get_bpe_settings(
        train_corpus,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=subdir_prefix,
        subword_nmt_repo_path=subword_nmt_repo,
    )
    return bpe_settings
