from typing import Tuple, Dict, Set, List, Optional, Union
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from .language_models.n_gram import get_count_based_n_gram
from .lm.ffnn import get_ffnn_lm
from .lm.trafo import get_trafo_lm
import re
from sisyphus import Job, Task, tk, gs
import i6_core.util as util

class GetBpeRatioJob(Job):
    """
    Apply BPE codes on a text file
    """

    __sis_hash_exclude__ = {"gzip_output": False}

    def __init__(
            self,
            text_file: tk.Path,
            vocab: str,
            subword_nmt_repo: Optional[tk.Path] = None,
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
        from .language_models.n_gram import ApplyBPEToTextJob
        self.text_file = text_file
        self.subword_nmt_repo = util.get_subword_nmt_repo(subword_nmt_repo)
        vocab = get_vocab_by_str(vocab)
        self.bpe_text  = ApplyBPEToTextJob(
            text_file=self.text_file,
            bpe_codes=vocab.codes,
            bpe_vocab=tk.Path(vocab.vocab.get_path() [:-5] + "dummy_count.vocab"), #
            subword_nmt_repo=self.subword_nmt_repo,
            gzip_output=True,
            mini_task=False,
        ).out_bpe_text
        self.out_ratio = self.output_var("bpe_to_word_ratio")

        self.mini_task = mini_task
        self.rqmt = {"cpu": 2, "mem": 4, "time": 12}

    def tasks(self):
        if self.mini_task:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        # Compute BPE-to-original token ratio:
        total_orig_tokens = 0
        total_bpe_tokens = 0
        with util.uopen(self.text_file, "rt") as fin, util.uopen(self.bpe_text, "rt") as fout:
            for orig_line, bpe_line in zip(fin, fout):
                orig_tokens = orig_line.strip().split()
                bpe_tokens = bpe_line.strip().split()
                total_orig_tokens += len(orig_tokens)
                total_bpe_tokens += len(bpe_tokens)

        # avoid division by zero
        if total_orig_tokens > 0:
            self.out_ratio.set(total_bpe_tokens / total_orig_tokens)
        else:
            self.out_ratio.set(1.0)  # fallback to 1.0 if no tokens

    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["mini_task"]
        return super().hash(parsed_args)

def build_ngram_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, bpe_ratio: Optional[float | tk.Variable]=None) -> Tuple[Dict, Dict, Set[str]]:
    print(f"start build ngram!")
    as_ckpt # noqa
    lms = {}
    ppl_results = {}
    lm_types = set()
    prune_threshs = [
                    #None,
                     #1e-9, #1.3e-8,
                     6.7e-8,
                     3e-7, #7e-7,
                     1.7e-6, 4.5e-6, 1e-5,
                     3e-5,
                     7e-4, #1.7e-3, 5e-3,
                     5e-2, #8e-2,1e-1, 3e-1,5e-1, 7e-1, 9e-1,
                     ]
    prune_threshs.reverse()
    prune_threshs_sub = [5e-2, None]
    for n_order in [2,
                    #4, 5,
                    6]:
        for train_fraction in [0.1, 0.15, 0.2, 0.25, 0.3,
                               #None,
                               ]:
            for prune_thresh in prune_threshs:
                if n_order in [1,2] and prune_thresh is not None:
                    if prune_thresh < 5e-2:
                        continue
                if n_order > 2 and prune_thresh is not None:
                    if prune_thresh > 5e-2:
                        continue
                if train_fraction is not None and prune_thresh not in prune_threshs_sub:
                    continue
                print(f"build ngram {n_order} thresh{prune_thresh}!")
                lm, ppl_log = get_count_based_n_gram(vocab, n_order, prune_thresh, train_fraction=train_fraction, word_ppl=word_ppl,bpe_ratio=bpe_ratio)
                lm_name = f"{n_order}gram_{vocab}" + (f"_fr{train_fraction}".replace(".","") if train_fraction is not None else "")
                if prune_thresh:
                    lm_name += f"_pr{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
                lms[lm_name] = lm
                ppl_results[lm_name] = ppl_log
                lm_types.add(f"{n_order}gram")
    print(f"build ngram {lms}!")
    return lms, ppl_results, lm_types

def build_word_ngram_lms(word_ppl: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    word_ppl # noqa
    lms = {}
    ppl_results = {}
    lm_types = set()
    for n_order in [4]:
        prune_threshs = [None]
        for prune_thresh in prune_threshs:
            lm, ppl_log = get_count_based_n_gram("word", n_order, prune_thresh)
            lm_name = f"{n_order}gram_word"
            if prune_thresh:
                lm_name += f"_{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
            lms[lm_name] = lm
            ppl_results[lm_name] = ppl_log
            lm_types.add(f"{n_order}gram")
    return lms, ppl_results, lm_types

def build_ffnn_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False,bpe_ratio: Optional[float | tk.Variable]=None) -> Tuple[Dict, Dict, Set[str]]:
    lms = {}
    ppl_results = {}
    lm_types = {"ffnn"}
    match = re.search(r"bpe(.+)", vocab)
    epochs = [5, 10, 20, 40, 50]
    lm_configs = {
        "std": {"context_size": 8, "num_layers": 2, "ff_hidden_dim": 2048, "dropout": 0.1},
        #"low": {"context_size": 5, "num_layers": 2, "ff_hidden_dim": 1024, "dropout": 0.2}
    }
    train_subsets = {"std": 80, "low": 30}
    for capa in lm_configs:
        config = lm_configs[capa]
        for checkpoint, ppl, epoch in get_ffnn_lm(get_vocab_by_str(vocab), **config,
                                                  word_ppl=word_ppl, epochs=epochs, train_subset=train_subsets[capa], bpe_ratio=bpe_ratio):
            config_ = config.copy()
            config_["class"] = "FeedForwardLm"
            name = f"ffnn{config['context_size']}_{epoch}{capa}_bpe{match.group(1)}"
            lms[name] = {
                "preload_from_files": {
                    "recog_lm": {
                        "prefix": "recog_language_model.",
                        "filename": checkpoint.checkpoint
                    },
                },
                "recog_language_model": config_
            } if not as_ckpt else checkpoint
            ppl_results[name] = ppl
    return lms, ppl_results, lm_types

def build_trafo_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, bpe_ratio: Optional[float | tk.Variable]=None) -> Tuple[Dict, Dict, Set[str]]:
    lms = {}
    ppl_results = {}
    lm_types = {"trafo"}
    match = re.search(r"bpe(.+)", vocab)
    config = {"num_layers": 12, "model_dim": 512, "dropout": 0.0, "class": "TransformerLm"}
    epochs = [20, 50]
    from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    bpe_data = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=int(match.group(1)))
    bpe = Bpe(dim=184, codes=bpe_data.bpe_codes, vocab=bpe_data.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")
    for checkpoint, ppl, epoch in get_trafo_lm(bpe, n_ep=50, bs_feat=10000, num_layers=12, word_ppl=word_ppl,
                                               model_dim=512, max_seqs=200, max_seq_length_default_target=True, epochs=epochs, bpe_ratio=bpe_ratio):
        name = f"trafo_{epoch}_bpe{match.group(1)}"
        lms[name] = {
            "preload_from_files": {
                "recog_lm": {
                    "prefix": "recog_language_model.",
                    "filename": checkpoint.checkpoint
                },
            },
            "recog_language_model": config
        } if not as_ckpt else checkpoint
        ppl_results[name] = ppl
    return lms, ppl_results, lm_types

def build_llms(word_ppl: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    from i6_experiments.users.zhang.experiments.lm.llm import get_llm
    lm_types = {"LLM"}
    llms, ppl_llms = get_llm(model_ids=["Llama-3.2-1B", "Llama-3.1-8B"], batch_sizes=[16, 4], prompt=None, word_ppl=word_ppl)
    return llms, ppl_llms, lm_types

def build_all_lms(vocab: str, lm_kinds: List[str] = None, as_ckpt: bool = False, word_ppl: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    lms, ppl, types = {}, {}, set()
    if lm_kinds is None:
        lm_kinds = {"ngram", "word_ngram", "ffnn", "trafo"}
    else:
        lm_kinds = set(lm_kinds)
    builders = {
        "ngram": build_ngram_lms,
        "word_ngram": build_word_ngram_lms,
        "ffnn": build_ffnn_lms,
        "trafo": build_trafo_lms,
        "LLM": build_llms,
    }
    bpe_ratio = None
    if word_ppl:
        from datasets.librispeech import get_librispeech_lm_combined_txt
        from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
        bpe_ratio = GetBpeRatioJob(get_librispeech_lm_combined_txt(), vocab, get_returnn_subword_nmt()).out_ratio
    for kind, builder in builders.items():
        if kind not in lm_kinds:
            continue
        try:
            l, p, t = builder(vocab, as_ckpt, word_ppl, bpe_ratio) if "vocab" in builder.__code__.co_varnames else builder(word_ppl=word_ppl)
            lms.update(l)
            ppl.update(p)
            types.update(t)
        except Exception:
            continue

    if not as_ckpt: # Consider move this to somewhere else
        lms.update({"NoLM": None})

    return lms, ppl, types
