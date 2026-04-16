"""
Language model helpers
"""
from sisyphus import tk, Task

from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence, List, Dict
import os
import subprocess as sp

from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.lm.kenlm import CompileKenLMJob

from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_core.lm.srilm import ComputeNgramLmPerplexityJob

from i6_experiments.users.zhang.datasets.vocab import ApplyBPEToTextJob
from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_lm_combined_txt, get_test_corpus_text
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

SRILM_PATH_APPTEK = tk.Path("/nas/models/asr/hzhang/tools/srilm-1.7.3/bin/i686-m64/")
SRILM_PATH_APPTEK.hash_overwrite = "APPTEK_DEFAULT_SRILM_PATH"

from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH as SRILM_PATH_I6

def get_4gram_binary_lm() -> Tuple[tk.Path, Dict[str, tk.Path | tk.Variable]]:
    """
    Returns the official LibriSpeech 4-gram ARPA LM

    :return: path to a binary LM file
    """
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    # lm_data = get_librispeech_lm_combined_txt()
    eval_lm_data_dict = dict()
    from i6_experiments.users.zhang.datasets.librispeech import get_test_corpus_text
    for key in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        eval_lm_data_dict[key] = get_test_corpus_text([key])
    arpa_lm = get_arpa_lm_dict()["4gram"]
    ppls = dict()
    ppl_scores = dict()
    for k, lm_eval_data in eval_lm_data_dict.items():
        ppl_job = ComputeNgramLmPerplexityJob(
            ngram_order=4,
            lm = arpa_lm, # Seems only accept arpa LM
            eval_data=lm_eval_data,
            ngram_exe=SRILM_PATH_APPTEK.join_right("ngram"),
            mem_rqmt=6,
            time_rqmt=1,
            extra_ppl_args= '-debug 2'
        )
        alias_name = f"ppl/LBS/4gram_official/{k}"
        tk.register_output(alias_name + "/ppl", ppl_job.out_ppl_log)
        ppls[k] = ppl_job.out_ppl_log
        ppl_scores[k] = ppl_job.out_ppl_score
    from i6_experiments.users.zhang.utils.report import ReportDictJob
    tk.register_output(f"ppl/LBS/4gram_official" + "/report",
                       ReportDictJob(outputs=ppl_scores).out_report_dict)
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=arpa_lm, kenlm_binary_folder=KENLM_BINARY_PATH).out_lm
    return arpa_4gram_binary_lm_job, ppl_scores
