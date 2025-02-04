"""
Language model helpers
"""
from sisyphus import tk, Task

from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence, List
import os
import subprocess as sp

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_core.lm.srilm import ComputeNgramLmPerplexityJob
from i6_experiments.users.zhang.experiments.language_models.n_gram import ApplyBPEToTextJob
from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_lm_combined_txt
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH

class CreateBinaryLMJob_(CreateBinaryLMJob):
    """
    Run the build_binary command of the KenLM toolkit to create a binary LM from an given ARPA LM
    """

    def __init__(
        self,
        *,
        arpa_lm: tk.Path,
        kenlm_binary_folder: tk.Path,
        quant_level: int
    ):
        """
        :param arpa_lm: any ARPA format LM
        :param kenlm_binary_folder: output of the CompileKenLMJob, or a direct link to the build
            dir of the KenLM repo
        """
        super().__init__(arpa_lm = arpa_lm, kenlm_binary_folder=kenlm_binary_folder)
        self.quant_level = quant_level

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        build_binary = os.path.join(self.kenlm_binary_folder.get_path(), "build_binary")
        sp.check_call([build_binary, "-q "+ str(self.quant_level), self.arpa_lm.get_path(), self.out_lm.get_path()])

def get_4gram_binary_lm(prunning = Optional[List[int]], quant_level = Optional[int]) -> Tuple[tk.Path, tk.Variable]:
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
    arpa_lm = get_arpa_lm_dict()["4gram"]
    ppl_job = ComputeNgramLmPerplexityJob(
        ngram_order=4,
        lm = arpa_lm, # Seems only accept arpa LM
        eval_data=lm_data, # This is train data for the LM. TODO: use same data for eval on ASR model
        ngram_exe=SRILM_PATH.join_right("ngram"),
        time_rqmt=1,
    )
    arpa_4gram_binary_lm_job = CreateBinaryLMJob_(
        arpa_lm=arpa_lm, kenlm_binary_folder=KENLM_BINARY_PATH, quant_level=quant_level
    )
    return arpa_4gram_binary_lm_job.out_lm, ppl_job.out_ppl_score
