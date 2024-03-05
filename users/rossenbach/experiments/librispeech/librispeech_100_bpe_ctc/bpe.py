import os.path
from functools import lru_cache

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.bpe import get_bpe_settings, get_returnn_subword_nmt


@lru_cache()
def get_librispeech_100h_bpe(bpe_size, unk_label="<unk>", output_prefix=""):
    """

    :param int bpe_size:
    :param str output_prefix
    :return:
    :rtype: BPESettings
    """

    output_prefix = os.path.join(output_prefix, "librispeech_100h_bpe_%i" % bpe_size)

    subword_nmt_commit_hash = "6ba4515d684393496502b79188be13af9cad66e2"
    subword_nmt_repo = get_returnn_subword_nmt(commit_hash=subword_nmt_commit_hash, output_prefix=output_prefix)
    train_clean_100 = get_bliss_corpus_dict("flac", "corpora")['train-clean-100']
    bpe_settings = get_bpe_settings(
        train_clean_100,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=output_prefix,
        subword_nmt_repo_path=subword_nmt_repo)
    return bpe_settings


