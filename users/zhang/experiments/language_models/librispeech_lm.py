"""
Language model helpers
"""
from sisyphus import tk

from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_core.lm.srilm import ComputeNgramLmPerplexityJob
from i6_experiments.users.zhang.experiments.language_models.n_gram import ApplyBPEToTextJob
from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_lm_combined_txt
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH

def get_4gram_binary_lm() -> Tuple[tk.Path, tk.Variable]:
    """
    Returns the official LibriSpeech 4-gram ARPA LM

    :return: path to a binary LM file
    """
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    lm_data = get_librispeech_lm_combined_txt()
    # bpe_text = ApplyBPEToTextJob(
    #     text_file=lm_data,
    #     bpe_codes=vocab.codes,
    #     bpe_vocab=tk.Path(vocab.vocab.get_path()[:-5] + "dummy_count.vocab"),
    #     subword_nmt_repo=subword_nmt,
    #     gzip_output=True,
    #     mini_task=False,
    # ).out_bpe_text
    ppl_job = ComputeNgramLmPerplexityJob(
        ngram_order=4,
        lm = get_arpa_lm_dict()["4gram"], # Seems only accept arpa LM
        eval_data=lm_data, # This is train data for the LM. TODO: use same data for eval on ASR model
        ngram_exe=SRILM_PATH.join_right("ngram"),
        time_rqmt=1,
    )
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=get_arpa_lm_dict()["4gram"], kenlm_binary_folder=KENLM_BINARY_PATH
    )
    return arpa_4gram_binary_lm_job.out_lm, ppl_job.out_ppl_score
