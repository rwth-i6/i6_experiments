import torch
from typing import Optional, Union, List
import os
import shutil
import sys
import tempfile
import subprocess as sp
import numpy as np

from i6_core.lm.kenlm import CompileKenLMJob, CreateBinaryLMJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.util import uopen
from i6_core.lib.lm import Lm
from i6_core.lm.srilm import ComputeNgramLmPerplexityJob
import i6_core.util as util

from sisyphus import Job, Task, tk, gs

from i6_experiments.users.mueller.datasets.librispeech import get_librispeech_lm_combined_txt, _extract_audio_seq_len_file, _extract_text_seq_len_file, _get_corpus_text_dict, TextDictToTextLinesJob
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

def get_binary_lm(arpa_path: tk.Path) -> tk.Path:
    """
    Returns a manually created LM

    :return: path to a binary LM file
    """
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    
    arpa_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=arpa_path, kenlm_binary_folder=KENLM_BINARY_PATH
    )
    return arpa_binary_lm_job.out_lm

def get_kenlm_n_gram(vocab: Bpe, N_order: int) -> tk.Path:
    assert vocab
    
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    
    subword_nmt = get_returnn_subword_nmt()
    
    lm_data = get_librispeech_lm_combined_txt()
    
    bpe_text = ApplyBPEToTextJob(
        text_file=lm_data,
        bpe_codes=vocab.codes,
        bpe_vocab=tk.Path(vocab.vocab.get_path()[:-5] + "dummy_count.vocab"),
        subword_nmt_repo=subword_nmt,
        gzip_output=True,
        mini_task=False,
    ).out_bpe_text
    
    lm_arpa = KenLMplzJob(
        text=[bpe_text],
        order=N_order,
        interpolate_unigrams=False,
        use_discount_fallback=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        pruning=None,
        vocabulary=None
    ).out_lm
    
    return lm_arpa

def get_count_based_n_gram(vocab: Bpe, N_order: int) -> tk.Path:    
    subword_nmt = get_returnn_subword_nmt()
    
    lm_arpa = get_kenlm_n_gram(vocab, N_order)
    
    for corpus_id in ["dev-other", "test-other", "train-other-960", "all"]:
        if corpus_id == "all":
            ppl_data = get_librispeech_lm_combined_txt()
        else:
            corpus = _get_corpus_text_dict(corpus_id)
            ppl_data = TextDictToTextLinesJob(corpus, gzip=True).out_text_lines
        ppl_text = ApplyBPEToTextJob(
            text_file=ppl_data,
            bpe_codes=vocab.codes,
            bpe_vocab=tk.Path(vocab.vocab.get_path()[:-5] + "dummy_count.vocab"),
            subword_nmt_repo=subword_nmt,
            gzip_output=True,
            mini_task=False,
        ).out_bpe_text
        ppl_job = ComputeNgramLmPerplexityJob(
            ngram_order=N_order,
            lm = lm_arpa,
            eval_data=ppl_text,
            ngram_exe=SRILM_PATH.join_right("ngram"),
            mem_rqmt=4,
            time_rqmt=1,
        )
        
        tk.register_output(f"datasets/LibriSpeech/lm/count_based_{N_order}-gram_{corpus_id}", ppl_job.out_ppl_score)
    
    conversion_job = ConvertARPAtoTensor(
        lm=lm_arpa,
        bpe_vocab=vocab.vocab,
        N_order=N_order,
    )
    
    conversion_job.add_alias(f"datasets/LibriSpeech/lm/count_based_{N_order}-gram")
        
    return conversion_job.out_lm_tensor

def get_prior_from_unigram(vocab: Bpe, prior_dataset: Optional[DatasetConfig], vocab_name: str) -> tk.Path:
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    KENLM_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_KENLM_BINARY_PATH"
    
    subword_nmt = get_returnn_subword_nmt()
    
    lm_data = get_librispeech_lm_combined_txt()
    
    bpe_text = ApplyBPEToTextJob(
        text_file=lm_data,
        bpe_codes=vocab.codes,
        bpe_vocab=tk.Path(vocab.vocab.get_path()[:-5] + "dummy_count.vocab"),
        subword_nmt_repo=subword_nmt,
        gzip_output=True,
        mini_task=False,
    ).out_bpe_text
    
    lm_arpa = KenLMplzJob(
        text=[bpe_text],
        order=1,
        interpolate_unigrams=False,
        use_discount_fallback=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        pruning=None,
        vocabulary=None
    ).out_lm
    
    ppl_job = ComputeNgramLmPerplexityJob(
        ngram_order=1,
        lm = lm_arpa,
        eval_data=bpe_text,
        ngram_exe=SRILM_PATH.join_right("ngram"),
        mem_rqmt=4,
        time_rqmt=1,
    )
    
    tk.register_output(f"datasets/LibriSpeech/lm/count_based_1-gram_all", ppl_job.out_ppl_score)
    
    bpe_len_wo_blank = _extract_text_seq_len_file(prior_dataset, vocab_name, name="target", use_main_ds=True)
    audio_len = _extract_audio_seq_len_file(prior_dataset, use_main_ds=True)
    prior_job = ExtractPrior(
        lm=lm_arpa,
        bpe_vocab=vocab.vocab,
        bpe_len_wo_blank=bpe_len_wo_blank,
        audio_len=audio_len,
    )
    
    prior_job.add_alias(f"datasets/LibriSpeech/lm/prior_from_unigram")
    tk.register_output(f"datasets/LibriSpeech/lm/prior_from_unigram", prior_job.out_prior_tensor)
    
    return prior_job.out_prior_tensor

#--------------------------------------------------------------

class ConvertARPAtoTensor(Job):
    def __init__(
        self,
        lm: tk.Path,
        bpe_vocab: tk.Path,
        N_order: int,
    ):
        self.lm = lm
        self.bpe_vocab = bpe_vocab
        self.N_order = N_order
        self.out_lm_tensor = self.output_path("lm.pt")

        # self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lm_loader = Lm(self.lm)
        
        vocab = eval(uopen(self.bpe_vocab, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(vocab, dict), "Has to be a dict containing the vocab!"
        vocab_n = len(vocab) - 1 # we combine eos and bos
        
        ret_tensor = None
        
        for N in reversed(range(2, self.N_order + 1)):
            n_grams = list(lm_loader.get_ngrams(N))
            
            # Read out the words and probabilities and turn into indexes of vocab
            n_grams = [list(map(lambda x: vocab[x], words.split(" "))) + [probs[0]] for words, probs in n_grams]
            n_grams = list(map(list, zip(*n_grams)))
            
            assert len(n_grams) - 1 == N, f"The conversion into a list failed ({len(n_grams) - 1} != {N})!"
            
            tensor = torch.full((vocab_n,)*N, float("-inf"), dtype=torch.float32)
            # Set the probabilites by using N indexes
            tensor[n_grams[:-1]] = torch.tensor(n_grams[-1], dtype=torch.float32)
            # The probs are in logs base 10
            tensor = torch.pow(10, tensor)
            
            atol = 0.005
            if self.N_order == 2:
                s = tensor.sum(1)
                assert s[0].allclose(torch.tensor(1.0), atol=atol), f"The next word probabilities for <s> do not sum to 1! {s[0]}"
                assert s[1].allclose(torch.tensor(0.0)), f"Prob of <unk> should be 0! (1) {s[1]}"
                assert s[2:].allclose(torch.tensor(1.0), atol=atol), f"The next word probabilities do not sum to 1! {s[2:]}"
            else:
                assert (tensor.sum(-1) < 1.005).all(), f"The next word probabilities are not smaller than 1! {tensor.sum(-1).flatten()[tensor.sum(-1).flatten() >= 1.005]}"
            assert tensor.sum(tuple(range(0, N-1)))[1].allclose(torch.tensor(0.0)), f"Prob of <unk> should be 0! (2) {tensor.sum(tuple(range(0, N-1)))[1]}"
            
            tensor = tensor.log()
            
            if ret_tensor is None:
                ret_tensor = tensor
            else:
                ret_tensor[*[0]*(self.N_order - N + 1)] = tensor[0]
            
        with uopen(self.out_lm_tensor, "wb") as f:
            torch.save(ret_tensor, f)
            
class ExtractPrior(Job):
    def __init__(
        self,
        lm: tk.Path,
        bpe_vocab: tk.Path,
        bpe_len_wo_blank: tk.Path,
        audio_len: tk.Path,
    ):
        self.lm = lm
        self.bpe_vocab = bpe_vocab
        self.out_prior_tensor = self.output_path("prior.txt")
        self.audio_len = audio_len
        self.bpe_len_wo_blank = bpe_len_wo_blank

        # self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lm_loader = Lm(self.lm)
        
        vocab = eval(uopen(self.bpe_vocab, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(vocab, dict), "Has to be a dict containing the vocab!"
        vocab_n = len(vocab) - 1 # we combine eos and bos
        
        bpe_len_wo_blank = np.loadtxt(self.bpe_len_wo_blank.get_path(), dtype="int32")
        audio_len = np.loadtxt(self.audio_len.get_path(), dtype="int32")
        
        assert len(bpe_len_wo_blank) == len(audio_len), "The lengths of the files do not match!"
        
        bpe_len_w_blank = np.ceil((audio_len + 1) / 960)
        
        bpe_len_w_blank = bpe_len_w_blank.sum()
        bpe_len_wo_blank = bpe_len_wo_blank.sum()
        
        uni_gram = list(lm_loader.get_ngrams(1))
        
        # Read out the words and probabilities and turn into indexes of vocab
        uni_gram = [list(map(lambda x: vocab[x], words.split(" "))) + [probs[0]] for words, probs in uni_gram]
        uni_gram = list(map(list, zip(*uni_gram)))
        
        assert len(uni_gram) - 1 == 1, f"The conversion into a list failed ({len(uni_gram) - 1} != {1})!"
        
        tensor = torch.full((vocab_n,), float("-inf"), dtype=torch.float32)
        # Set the probabilites by using N indexes
        tensor[uni_gram[:-1]] = torch.tensor(uni_gram[-1], dtype=torch.float32)
        # The probs are in logs base 10
        tensor = torch.pow(10, tensor)
        
        atol = 0.005
        assert tensor[1].allclose(torch.tensor(0.0), atol=0.0001), f"Prob of <unk> should be 0! (1) {tensor[1]}"
        assert tensor.sum().allclose(torch.tensor(1.0), atol=atol), f"The word probabilities do not sum to 1! {tensor.sum()}"
        
        tensor = tensor * bpe_len_wo_blank
        
        tensor = torch.cat([tensor, torch.tensor([bpe_len_w_blank - bpe_len_wo_blank], dtype=torch.float32)])
        tensor = tensor / bpe_len_w_blank
        
        assert tensor.sum().allclose(torch.tensor(1.0), atol=atol), f"The word probabilities do not sum to 1! {tensor.sum()}"
        
        tensor = tensor.log()
        tensor = tensor.numpy()
        
        with uopen(self.out_prior_tensor, "w") as f:
            np.savetxt(f, tensor, delimiter=" ")
            
class ApplyBPEToTextJob(Job):
    """
    Apply BPE codes on a text file
    """

    __sis_hash_exclude__ = {"gzip_output": False}

    def __init__(
        self,
        text_file: tk.Path,
        bpe_codes: tk.Path,
        bpe_vocab: Optional[tk.Path] = None,
        subword_nmt_repo: Optional[tk.Path] = None,
        gzip_output: bool = False,
        mini_task=True,
    ):
        """
        :param text_file: words text file to convert to bpe
        :param bpe_codes: bpe codes file, e.g. ReturnnTrainBpeJob.out_bpe_codes
        :param bpe_vocab: if provided, then merge operations that produce OOV are reverted,
            use e.g. ReturnnTrainBpeJob.out_bpe_dummy_count_vocab
        :param subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        :param gzip_output: use gzip on the output text
        :param mini_task: if the Job should run locally, e.g. only a small (<1M lines) text should be processed
        """
        self.text_file = text_file
        self.bpe_codes = bpe_codes
        self.bpe_vocab = bpe_vocab
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        self.gzip_output = gzip_output

        self.out_bpe_text = self.output_path("words_to_bpe.txt.gz" if gzip_output else "words_to_bpe.txt")

        self.mini_task = mini_task
        self.rqmt = {"cpu": 2, "mem": 4, "time": 12}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            input_file = self.text_file.get_path()
            tmp_infile = os.path.join(tmp, "in_text.txt")
            tmp_outfile = os.path.join(tmp, "out_text.txt")
            with util.uopen(tmp_infile, "wt") as out:
                sp.call(["zcat", "-f", input_file], stdout=out)
            cmd = [
                sys.executable,
                os.path.join(self.subword_nmt_repo.get_path(), "apply_bpe.py"),
                "--input",
                tmp_infile,
                "--codes",
                self.bpe_codes.get_path(),
                "--output",
                tmp_outfile,
            ]

            if self.bpe_vocab:
                cmd += ["--vocabulary", self.bpe_vocab.get_path()]
                
            util.create_executable("apply_bpe.sh", cmd)
            sp.run(cmd, check=True)

            if self.gzip_output:
                with util.uopen(tmp_outfile, "rt") as fin, util.uopen(self.out_bpe_text, "wb") as fout:
                    sp.call(["gzip"], stdin=fin, stdout=fout)
            else:
                shutil.copy(tmp_outfile, self.out_bpe_text.get_path())

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)
    
class KenLMplzJob(Job):
    """
    Run the lmplz command of the KenLM toolkit to create a gzip compressed ARPA-LM file
    """

    def __init__(
        self,
        *,
        text: Union[tk.Path, List[tk.Path]],
        order: int,
        interpolate_unigrams: bool,
        use_discount_fallback: bool,
        pruning: Optional[List[int]],
        vocabulary: Optional[tk.Path],
        kenlm_binary_folder: tk.Path,
        mem: float = 4.0,
        time: float = 1.0,
    ):
        """

        :param text: training text data
        :param order: "N"-order of the "N"-gram LM
        :param interpolate_unigrams: Set True for KenLM default, and False for SRILM-compatibility.
            Having this as False will increase the share of the unknown probability
        :param pruning: absolute pruning threshold for each order,
            e.g. to remove 3-gram and 4-gram singletons in a 4th order model use [0, 0, 1, 1]
        :param vocabulary: a "single word per line" file to determine valid words,
            everything else will be treated as unknown
        :param kenlm_binary_folder: output of the CompileKenLMJob, or a direct link to the build
            dir of the KenLM repo
        :param mem: memory rqmt, needs adjustment for large training corpora
        :param time: time rqmt, might adjustment for very large training corpora and slow machines
        """
        self.text = text
        self.order = order
        self.interpolate_unigrams = interpolate_unigrams
        self.pruning = pruning
        self.vocabulary = vocabulary
        self.kenlm_binary_folder = kenlm_binary_folder
        self.use_discount_fallback = use_discount_fallback

        self.out_lm = self.output_path("lm.gz")

        self.rqmt = {"cpu": 1, "mem": mem, "time": time}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp:
            lmplz_command = [
                os.path.join(self.kenlm_binary_folder.get_path(), "lmplz"),
                "-o",
                str(self.order),
                "--interpolate_unigrams",
                str(self.interpolate_unigrams),
                "-S",
                "%dG" % int(self.rqmt["mem"]),
                "-T",
                tmp,
            ]
            if self.use_discount_fallback:
                lmplz_command += ["--discount_fallback"]
            if self.pruning is not None:
                lmplz_command += ["--prune"] + [str(p) for p in self.pruning]
            if self.vocabulary is not None:
                lmplz_command += ["--limit_vocab_file", self.vocabulary.get_path()]

            zcat_command = ["zcat", "-f"] + [text.get_path() for text in self.text]
            with uopen(self.out_lm, "wb") as lm_file:
                p1 = sp.Popen(zcat_command, stdout=sp.PIPE)
                p2 = sp.Popen(lmplz_command, stdin=p1.stdout, stdout=sp.PIPE)
                sp.check_call("gzip", stdin=p2.stdout, stdout=lm_file)
                if p2.returncode:
                    raise sp.CalledProcessError(p2.returncode, cmd=lmplz_command)

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mem"]
        del parsed_args["time"]
        return super().hash(parsed_args)