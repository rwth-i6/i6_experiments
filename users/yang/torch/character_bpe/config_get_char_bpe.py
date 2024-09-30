from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection



from sisyphus import tk
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob

_sis_prefix: Optional[str] = None
def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def sis_run_with_prefix(prefix_name: Optional[str] = None):

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    # dummy bpe training task, to get character level vocab
    corpus = tk.Path("/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/corpus/convert/CorpusToTxtJob.4gG5aKOKTRa5/output/corpus.txt")
    # from Nick's 10k bpe setup
    bpe_size = 1
    unk_label = "<unk>"
    subword_nmt_repo = tk.Path("/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/tools/git/CloneGitRepositoryJob.rEnqw3X2cgyB/output/subword-nmt")
    bpe_job = ReturnnTrainBpeJob(text_file=corpus, bpe_size=bpe_size, unk_label=unk_label, subword_nmt_repo=subword_nmt_repo)
    tk.register_output(
        _sis_prefix + '/bpe_vocab.txt',
        bpe_job.out_bpe_vocab
    )
    tk.register_output(
        _sis_prefix + '/dummy_bpe_code.txt',
        bpe_job.out_bpe_codes
    )

py = sis_run_with_prefix
