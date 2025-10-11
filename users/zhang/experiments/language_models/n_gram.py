from __future__ import annotations

#-----------------------------------------
import torch
from typing import Optional, Union, List, Any
import os
import shutil
import sys
import tempfile
import subprocess as sp
import re

from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence

from i6_experiments.users.zhang.datasets.vocab import GetSubwordRatioJob
from i6_core.lm.kenlm import CompileKenLMJob#, CreateBinaryLMJob
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.util import uopen, create_executable, relink
from i6_core.lib.lm import Lm
from i6_core.lm.srilm import ComputeNgramLmPerplexityJob
import i6_core.util as util
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from sisyphus.job_path import Path

from sisyphus import Job, Task, tk, gs

from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_lm_combined_txt, _get_test_corpus_text, \
    _extract_audio_seq_len_file, _extract_text_seq_len_file, get_test_corpus_text
from i6_experiments.users.zhang.datasets.vocab import ApplyBPEToTextJob
from i6_experiments.users.zhang.utils.report import ReportDictJob
from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.baselines.tedlium2.default_tools import SRILM_PATH as LBS_SRILM_PATH
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig

rqmt_map = {4: [("mem", 15),("time", 2)], 5: [("mem", 20),("time", 2)], 6: [("mem", 20),("time", 2)],  # Compare to bpe 128 Need much more for bpe10k and more for whole word
                                             7: [("mem", 25),("time", 2)],
                                                 8: [("mem", 30),("time", 2)],
            9: [("mem", 35),("time", 2)],
            10: [("mem", 40),("time", 2)]} # rqmt_map for n_gram KenLMplz job. Smaller as 4 use default setting
default_rqmt = [("mem", 12),("time", 1)]

SRILM_PATH_APPTEK = tk.Path("/nas/models/asr/hzhang/tools/srilm-1.7.3/bin/i686-m64/")
SRILM_PATH_APPTEK.hash_overwrite = "APPTEK_SPAINISH_DEFAULT_SRILM_PATH"
def py():
    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab
    _, spm, _ = get_model_and_vocab()
    vocab_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])
    task_name = "ES"
    for N_order in [4,5,6]:
        get_count_based_n_gram(vocab_config, N_order,task_name=task_name, word_ppl=True, only_transcription=True)


def get_count_based_n_gram(vocab: [str | VocabConfig], N_order: int, prune_thresh: Optional[float]=None, task_name: str = "LBS", train_fraction: float = None, word_ppl: bool =False, only_transcription: bool = False, eval_keys: set = None) -> \
tuple[Path, dict[str | Any, Path | Any]]:
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    hash_name = "LBS" if task_name == "LIBRISPEECH" else task_name
    KENLM_BINARY_PATH.hash_overwrite = f"{hash_name}_DEFAULT_KENLM_BINARY_PATH"
    vocab_str = vocab if isinstance(vocab, str) else "ES_spm10k" # For LBS vocab_config can be get with str
    if N_order >= 4 and vocab != "bpe128":
        rqmt = dict(rqmt_map[N_order])
    else:
        rqmt = dict(default_rqmt)
    subword_nmt = get_returnn_subword_nmt()
    eval_lm_data_dict = dict()
    if task_name == "LBS":
        lm_data = get_librispeech_lm_combined_txt()
        for key in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            eval_lm_data_dict[key] = get_test_corpus_text([key])
    elif task_name == "ES":
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.lm.data import LM_TRAIN_DATA, \
            LM_TRANS_TRAIN_DATA
        from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import get_lm_eval_text

        if only_transcription:
            lm_data = LM_TRANS_TRAIN_DATA
        else:
            from i6_core.text.processing import ConcatenateJob
            lm_data = ConcatenateJob([LM_TRANS_TRAIN_DATA, LM_TRAIN_DATA]).out
        for key in DEV_KEYS + TEST_KEYS:
            eval_lm_data_dict[key] = get_lm_eval_text(key=key)
    else:
        raise ValueError("Unknown task name {}".format(task_name))

    if train_fraction and train_fraction < 1.0:
        lm_data = ReduceCorpusByTokenFractionJob(lm_data, target_fraction=train_fraction).out
    if re.match("^bpe[0-9]+.*$", vocab_str):
        if task_name == "LBS":
            from ..language_models.librispeech import _get_bpe_vocab, bpe10k
            if vocab_str == "bpe10k":
                vocab = bpe10k
                rqmt = dict([(key, value*4) for key, value in rqmt.items()])
            else:
                vocab = _get_bpe_vocab(bpe_size=vocab[len("bpe") :])
        text = ApplyBPEToTextJob(
            text_file=lm_data,
            bpe_codes=vocab.codes,
            bpe_vocab=tk.Path(vocab.vocab.get_path() [:-5] + "dummy_count.vocab"), #
            subword_nmt_repo=subword_nmt,
            gzip_output=True,
            mini_task=False,
        ).out_bpe_text
        for k, orig_text in eval_lm_data_dict.items():
            eval_lm_data_dict[k] = ApplyBPEToTextJob(
                text_file=orig_text,
                bpe_codes=vocab.codes,
                bpe_vocab=tk.Path(vocab.vocab.get_path() [:-5] + "dummy_count.vocab"), #
                subword_nmt_repo=subword_nmt,
                gzip_output=True,
                mini_task=False,
            ).out_bpe_text
    elif "spm10k" in vocab_str and task_name == "ES":
        assert isinstance(vocab, SentencePieceModel)
        text = ApplySentencepieceToTextJob(
            text_file=lm_data,
            sentencepiece_model=vocab.model_file,
            gzip_output=True,
            enable_unk=False,
        ).out_sentencepiece_text
        for k, orig_text in eval_lm_data_dict.items():
            eval_lm_data_dict[k] = ApplySentencepieceToTextJob(
                text_file=orig_text,
                sentencepiece_model=vocab.model_file,
                gzip_output=True,
                enable_unk=False,
            ).out_sentencepiece_text
    else: # Train on whole word
        text = lm_data
        rqmt = dict([(key, value * 5) for key, value in rqmt.items()])
    lm_arpa = KenLMplzJob(
        text=[text],
        order=N_order,
        interpolate_unigrams=False, # Set false for Compatibility with srilm
        use_discount_fallback=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        pruning=[0, 0, 0] + [1 for _ in range(N_order-3)] if N_order > 2 else None, # 3e-7
        vocabulary=None,
        **rqmt
    ).out_lm
    SRILM_PATH = LBS_SRILM_PATH if task_name == "LIBRISPEECH" else SRILM_PATH_APPTEK
    if prune_thresh:
        lm_arpa = PruneLMJob(N_order, lm_arpa, prune_thresh, SRILM_PATH.join_right("ngram")).out_lm
        arpa_binary_lm = CreateBinaryLMJob(
            arpa_lm=lm_arpa, kenlm_binary_folder=KENLM_BINARY_PATH, probing_multiplier=2.0 if prune_thresh > 4.2e-6 else None,**rqmt
        ).out_lm
    else:
        arpa_binary_lm = CreateBinaryLMJob(
            arpa_lm=lm_arpa, kenlm_binary_folder=KENLM_BINARY_PATH, **rqmt
        ).out_lm
    ppls = dict()
    ppl_scores = dict()
    for k, lm_eval_data in eval_lm_data_dict.items():
        if eval_keys and k not in eval_keys:
            continue
        ppl_job = ComputeNgramLmPerplexityJob(
            ngram_order=N_order,
            lm = lm_arpa, # Seems only accept arpa LM
            eval_data=lm_eval_data,
            ngram_exe=SRILM_PATH.join_right("ngram"),
            mem_rqmt=rqmt["mem"] + 10 if N_order == 6 else rqmt["mem"],
            time_rqmt=1,
            extra_ppl_args= '-debug 2'
        )
        if isinstance(vocab, SentencePieceModel) or "spm" in vocab_str:
            apply_job = ApplySentencepieceToTextJob
        elif isinstance(vocab, Bpe) or "bpe" in vocab_str:
            apply_job = ApplyBPEToTextJob
        else:
            assert vocab == "word", "vocab must be SentencePieceModel or SentencePieceModel, or word"
            apply_job = None
        if apply_job:
            ratio = GetSubwordRatioJob(lm_eval_data, vocab, get_returnn_subword_nmt(),apply_job=apply_job).out_ratio
            tk.register_output(f"{task_name}_{k}_{vocab_str}_ratio", ratio)
        else:
            ratio = 1
        if word_ppl and vocab != "word":
            ppl_job = PPLConvertJob(ppl_job.out_ppl_log, ratio)
        alias_name = f"ppl/{task_name}/{N_order}gram_{prune_thresh}" + ('_'+ str(train_fraction) if train_fraction else '') + f"/{k}" + ("word_ppl" if word_ppl else "")
        #tk.register_output(alias_name + "/ppl", ppl_job.out_ppl_log)
        ppls[k] = ppl_job.out_ppl_log
        ppl_scores[k] = ppl_job.out_ppl_score
    tk.register_output(f"ppl/{task_name}/{N_order}gram_{prune_thresh}" + ('_'+ str(train_fraction) if train_fraction else '') + "/" + ("word_ppl" if word_ppl else "") + "report",
                       ReportDictJob(outputs=ppl_scores).out_report_dict)
    # conversion_job = ConvertARPAtoTensor(
    #     lm=lm_arpa,
    #     bpe_vocab=vocab.vocab,
    #     N_order=N_order,
    # )
    #
    # conversion_job.add_alias(f"datasets/LibriSpeech/lm/count_based_{N_order}-gram")
        
    #return conversion_job.out_lm_tensor
    return arpa_binary_lm, ppl_scores

def get_apptek_ES_n_gram(vocab: [str | VocabConfig], N_order: int, prune_thresh: Optional[float], task_name: str = "ES", train_fraction: float = None, word_ppl: bool =False) -> \
tuple[Path, dict[str | Any, Path | Any]]:
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import get_lm_eval_text
    assert task_name == "ES"
    lm_file_path = "/nas/models/asr/artefacts/lm/ES/20220905-wwang-srilm-tel/es.tel.20220818.4-gram.lm.arpa.gz"
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm").out_repository.copy()
    KENLM_BINARY_PATH = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    hash_name = "LBS" if task_name == "LIBRISPEECH" else task_name
    KENLM_BINARY_PATH.hash_overwrite = f"{hash_name}_DEFAULT_KENLM_BINARY_PATH"
    vocab_str = vocab if isinstance(vocab, str) else "ES_spm10k" # For LBS vocab_config can be get with str
    if N_order > 4 and vocab != "bpe128":
        rqmt = dict(rqmt_map[N_order])
    else:
        rqmt = dict(default_rqmt)
    subword_nmt = get_returnn_subword_nmt()
    eval_lm_data_dict = dict()
    for key in DEV_KEYS + TEST_KEYS:
        eval_lm_data_dict[key] = get_lm_eval_text(key=key)

    if re.match("^bpe[0-9]+.*$", vocab_str):
        for k, orig_text in eval_lm_data_dict.items():
            eval_lm_data_dict[k] = ApplyBPEToTextJob(
                text_file=orig_text,
                bpe_codes=vocab.codes,
                bpe_vocab=tk.Path(vocab.vocab.get_path() [:-5] + "dummy_count.vocab"), #
                subword_nmt_repo=subword_nmt,
                gzip_output=True,
                mini_task=False,
            ).out_bpe_text
    elif "spm10k" in vocab_str and task_name == "ES":
        assert isinstance(vocab, SentencePieceModel)
        for k, orig_text in eval_lm_data_dict.items():
            eval_lm_data_dict[k] = ApplySentencepieceToTextJob(
                text_file=orig_text,
                sentencepiece_model=vocab.model_file,
                gzip_output=True,
                enable_unk=False,
            ).out_sentencepiece_text
    else: # Train on whole word
        rqmt = dict([(key, value * 5) for key, value in rqmt.items()])
    lm_arpa = tk.Path(lm_file_path)
    SRILM_PATH = LBS_SRILM_PATH if task_name == "LIBRISPEECH" else SRILM_PATH_APPTEK
    if prune_thresh:
        lm_arpa = PruneLMJob(N_order, lm_arpa, prune_thresh, SRILM_PATH.join_right("ngram")).out_lm
        arpa_binary_lm = CreateBinaryLMJob(
            arpa_lm=lm_arpa, kenlm_binary_folder=KENLM_BINARY_PATH, probing_multiplier=2.0 if prune_thresh > 4.2e-6 else None,**rqmt
        ).out_lm
    else:
        arpa_binary_lm = CreateBinaryLMJob(
            arpa_lm=lm_arpa, kenlm_binary_folder=KENLM_BINARY_PATH, **rqmt
        ).out_lm
    ppls = dict()
    for k, lm_eval_data in eval_lm_data_dict.items():
        ppl_job = ComputeNgramLmPerplexityJob(
            ngram_order=N_order,
            lm = lm_arpa, # Seems only accept arpa LM
            eval_data=lm_eval_data, # This is train data for the LM.
            ngram_exe=SRILM_PATH.join_right("ngram"),
            mem_rqmt=rqmt["mem"],
            time_rqmt=1,
            extra_ppl_args= '-debug 2'
        )
        if isinstance(vocab, SentencePieceModel) or "spm" in vocab_str:
            apply_job = ApplySentencepieceToTextJob
        elif isinstance(vocab, Bpe) or "bpe" in vocab_str:
            apply_job = ApplyBPEToTextJob
        else:
            assert vocab == "word", "vocab must be SentencePieceModel or SentencePieceModel, or word"
            apply_job = None
        if apply_job:
            ratio = GetSubwordRatioJob(lm_eval_data, vocab, get_returnn_subword_nmt(),apply_job=apply_job).out_ratio
            tk.register_output(f"{task_name}_{k}_{vocab_str}_ratio", ratio)
        else:
            ratio = 1
        if word_ppl and vocab != "word":
            ppl_job = PPLConvertJob(ppl_job.out_ppl_log, ratio)
        alias_name = f"ppl/{task_name}/{N_order}gram{vocab}_apptek_tel_{prune_thresh if prune_thresh else ''}" + ('_'+ str(train_fraction) if train_fraction else '') + f"/{k}" + ("word_ppl" if word_ppl else "")
        #tk.register_output(alias_name + "/ppl", ppl_job.out_ppl_log)
        ppls[k] = ppl_job.out_ppl_log
    # conversion_job = ConvertARPAtoTensor(
    #     lm=lm_arpa,
    #     bpe_vocab=vocab.vocab,
    #     N_order=N_order,
    # )
    #
    # conversion_job.add_alias(f"datasets/LibriSpeech/lm/count_based_{N_order}-gram")

    #return conversion_job.out_lm_tensor
    return arpa_binary_lm, ppls



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
        lm=lm_arpa,
        eval_data=bpe_text,
        ngram_exe=SRILM_PATH.join_right("ngram"),
        mem_rqmt=4,
        time_rqmt=1,
    )

    tk.register_output(f"datasets/LibriSpeech/lm/count_based_1-gram", ppl_job.out_ppl_score)

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

class PPLConvertJob(Job):
    def __init__(self, ppl:tk.Path, exponent: Union[float,tk.Variable]):
        self.original_ppl = ppl
        self.out_ppl_log = ppl
        self.exponent = exponent
        self.out_ppl_score = self.output_var("perplexity.score")
        self.out_num_sentences = self.output_var("num_sentences")
        self.out_num_words = self.output_var("num_words")
        self.out_num_oovs = self.output_var("num_oovs")

    def tasks(self):
        #yield Task("run", mini_task=True)
        yield Task("get_ppl", mini_task=True)

    def get_ppl(self):
        """extracts various outputs from the ppl.log file"""
        from collections import deque
        exponent = self.exponent.get() if isinstance(self.exponent, tk.Variable) else self.exponent
        with open(self.original_ppl.get_path(), "rt") as f:
            lines = deque(f, maxlen=2)
            for line in lines:
                line = line.split(" ")
                for idx, ln in enumerate(line):
                    if ln == "sentences,":
                        self.out_num_sentences.set(int(line[idx - 1]))
                    if ln == "words,":
                        self.out_num_words.set(int(float(line[idx - 1])))
                    if ln == "OOVs":
                        self.out_num_oovs.set(int(line[idx - 1]))
                    if ln == "ppl=":
                        self.out_ppl_score.set(float(line[idx + 1])**exponent)

    # def run(self):
    #     with open(self.original_ppl.get_path(), "rt") as f:
    #         lines = f.readlines()[-2:]
    #         for line in lines:
    #             line = line.split(" ")
    #             for idx, ln in enumerate(line):
    #                 if ln == "ppl=" or ln == "Perplexity:":
    #                     ppl = float(line[idx + 1])
    #     with open(self.out_ppl_log.get_path(), "wt") as f:
    #         f.write(f"ppl={ppl**self.exponent}\n")

class ReduceCorpusByTokenFractionJob(Job):
    # TODO: add a random seed
    """
    Reduces a text corpus to a given percentage of total tokens (based on space-separated tokens).
    Output will be plain or gzipped depending on `zip_out`.
    """

    def __init__(self, input_file: tk.Path, target_fraction: float, zip_out: bool = True):
        """
        :param input_file: path to the input text corpus (raw or gz)
        :param target_fraction: target fraction of total tokens to keep (e.g., 0.10 for 10%)
        :param zip_out: whether to gzip the output
        :param out_name: name prefix for the output file
        """
        assert 0 < target_fraction <= 1.0, "target_fraction must be in (0, 1]"
        assert isinstance(input_file, tk.Path)

        self.input_file = input_file
        self.target_fraction = target_fraction
        self.zip_out = zip_out

        self.out = self.output_path("reduced_train.txt" + (".gz" if zip_out else ""))

    def tasks(self):
        yield Task("run", rqmt={"mem": 3, "time": 3})

    def run(self):
        import gzip
        import random
        # Resolve path and detect gzip
        in_path = gs.file_caching(self.input_file) if isinstance(self.input_file,
                                                                 str) else self.input_file.get_cached_path()
        is_gz = str(in_path).endswith('.gz')
        open_in = gzip.open if is_gz else open

        # First pass: count total tokens
        print("Counting total tokens...\n")
        total_tokens = 0
        with open_in(in_path, 'rt', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                total_tokens += len(stripped.split())

        target_tokens = int(total_tokens * self.target_fraction)
        remaining_tokens = total_tokens
        selected_tokens = 0
        selected_lines = 0

        # Prepare output stream
        print(f"Try to picking {self.target_fraction*100}% from original corpus...\n")
        mode = 'wt'
        open_out = gzip.open if self.zip_out else open
        with open_out(self.out, mode, encoding='utf-8') as fo:
            # Second pass: reservoir-style weighted sampling by tokens
            with open_in(in_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    ntoks = len(stripped.split())
                    if remaining_tokens <= 0:
                        break
                    # Probability of selecting this line
                    prob = (target_tokens - selected_tokens) / remaining_tokens
                    prob = min(max(prob, 0.0), 1.0)
                    if random.random() < prob:
                        fo.write(line)
                        selected_tokens += ntoks
                        selected_lines += 1
                        if selected_tokens >= target_tokens:
                            break
                    remaining_tokens -= ntoks

        print(f"Wrote {selected_lines} lines, approx. {selected_tokens} tokens ({self.target_fraction:.2%}).")

class PruneLMJob(Job):
    """
    Job that prunes the given LM
    """

    def __init__(
        self,
        ngram_order: int,
        lm: tk.Path,
        prune_thresh: float,
        ngram_exe: tk.Path,
        *,
        mem_rqmt: int = 48,
        time_rqmt: float = 3,
        cpu_rqmt: int = 1,
        fs_rqmt: str = "100G",
    ):
        """

        :param ngram_order: Maximum n gram order
        :param lm: LM to be pruned
        :param prune_thresh: Pruning threshold
        :param ngram_exe: Path to srilm ngram-count executable
        :param mem_rqmt: Memory requirements of Job (not hashed)
        :param time_rqmt: Time requirements of Job (not hashed)
        :param cpu_rqmt: Amount of Cpus required for Job (not hashed)
        :param fs_rqmt: Space on fileserver required for Job, example: "200G" (not hashed)
        """

        self.ngram_order = ngram_order
        self.lm = lm
        self.prune_thresh = prune_thresh
        self.ngram_exe = ngram_exe

        self.out_lm = self.output_path("pruned_lm.gz")

        self.rqmt_run = {
            "mem": mem_rqmt,
            "time": time_rqmt,
            "cpu": cpu_rqmt,
            "qsub_args": f"-l h_fsize={fs_rqmt}",
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt_run)

    def create_files(self):
        """creates bash script that will be executed in the run Task"""
        cmd = [
            f"{self.ngram_exe} \\\n",
            f"  -order {self.ngram_order} \\\n",
            f"  -renorm -unk \\\n",
            f"  -lm {self.lm.get_path()} \\\n",
            f"  -write-lm pruned.lm.gz \\\n",
            #f"  -preserve-vocab \\\n" if self.prune_thresh > 1e-5 else "",
            f"  -prune {self.prune_thresh} \\\n",
            f"  -memuse \n",
        ]
        create_executable("run.sh", cmd)

    def run(self):
        """executes the previously created script and relinks the lm from work folder to output folder"""
        sp.check_call("./run.sh")
        relink("pruned.lm.gz", self.out_lm.get_path())

    @classmethod
    def hash(cls, kwargs):
        """delete the queue requirements from the hashing"""
        del kwargs["mem_rqmt"]
        del kwargs["cpu_rqmt"]
        del kwargs["time_rqmt"]
        del kwargs["fs_rqmt"]
        return super().hash(kwargs)

# Borrowed, modiefied rqmt
class CreateBinaryLMJob(Job):
    """
    Run the build_binary command of the KenLM toolkit to create a binary LM from an given ARPA LM
    """
    __sis_hash_exclude__ = {"probing_multiplier" : None}
    def __init__(
        self,
        *,
        arpa_lm: tk.Path,
        kenlm_binary_folder: tk.Path,
        mem: float = 4.0,
        time: float = 1.0,
        probing_multiplier: float = None,
    ):
        """
        :param arpa_lm: any ARPA format LM
        :param kenlm_binary_folder: output of the CompileKenLMJob, or a direct link to the build
            dir of the KenLM repo
        """
        self.arpa_lm = arpa_lm
        self.kenlm_binary_folder = kenlm_binary_folder
        self.probing_multiplier = f"-p {probing_multiplier}" if probing_multiplier else ""
        self.out_lm = self.output_path("lm.bin")

        self.rqmt = {"cpu": 1, "mem": mem, "time": time}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        build_binary = os.path.join(self.kenlm_binary_folder.get_path(), "build_binary")
        if self.probing_multiplier:
            sp.check_call([build_binary, self.probing_multiplier, self.arpa_lm.get_path(), self.out_lm.get_path()])
        else:
            sp.check_call([build_binary, self.arpa_lm.get_path(), self.out_lm.get_path()])

    # @classmethod
    # def hash(cls, kwargs):
    #     """delete the queue requirements from the hashing"""
    #     del kwargs["mem"]
    #     del kwargs["time"]
    #     return super().hash(kwargs)

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
                assert (tensor.sum(-1) < 1.0).all(), f"The next word probabilities are not smaller than 1! {tensor.sum(-1)}"
            assert tensor.sum(tuple(range(0, N-1)))[1].allclose(torch.tensor(0.0)), f"Prob of <unk> should be 0! (2) {tensor.sum(tuple(range(0, N-1)))[1]}"
            
            tensor = tensor.log()
            
            if ret_tensor is None:
                ret_tensor = tensor
            else:
                #ret_tensor[*[0]*(self.N_order - N + 1)] = tensor[0]
                ret_tensor[tuple([0] * (self.N_order - N + 1))] = tensor[0]
        with uopen(self.out_lm_tensor, "wb") as f:
            torch.save(ret_tensor, f)
            

    
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
        yield Task("run", rqmt=self.rqmt)

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
            lmplz_command += ["--skip_symbols"]

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

def py():
    # from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab, \
    #     NETWORK_CONFIG_KWARGS
    from i6_experiments.users.zhang.experiments.lm.trafo import get_ES_trafo
    # _, spm, _ = get_model_and_vocab(fine_tuned_model=True)
    # vocab_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])

    #get_count_based_n_gram(vocab=vocab_config, N_order=4, task_name="ES", word_ppl=True)
    get_ES_trafo(epochs=[100],word_ppl=True,only_transcript=False,old=False)