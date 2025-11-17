import functools
from typing import Tuple, Dict, Set, List, Optional, Union, Type

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from returnn_common.datasets_old_2022_10.interface import VocabConfig

from .language_models.n_gram import get_count_based_n_gram
from .lm.ffnn import get_ffnn_lm
from .lm.trafo import get_trafo_lm
import re
from sisyphus import Job, Task, tk, gs

from functools import lru_cache
from collections import namedtuple


def build_ngram_lm(vocab: [str | VocabConfig], as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False,
                   task_name: str = "LBS", only_transcript: bool = False, n_order:int = 4, prune_thresh: float=None, fraction: float=None, eval_keys: set = None)-> Tuple[Dict, Dict, Set[str]]:
    vocab_str = vocab if isinstance(vocab, str) else ""
    if isinstance(vocab, VocabConfig):
        assert isinstance(vocab, SentencePieceModel) and vocab.dim == 10240
        vocab_str = "spm10k"
    as_ckpt # noqa
    lms = {}
    ppl_results = {}
    lm_types = set()
    lm, ppl_log = get_count_based_n_gram(vocab, n_order, word_ppl=word_ppl,task_name=task_name, only_transcription=only_transcript,prune_thresh=prune_thresh,train_fraction=fraction, eval_keys=eval_keys)
    lm_name = (f"{n_order}gram_{vocab_str}" + f"{'_trans' if only_transcript else ''}"
               + (f"_fr{fraction}".replace(".","") if fraction is not None else "")
               + (f"_pr{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_") if prune_thresh is not None else "")
              )
    lms[lm_name] = lm
    ppl_results[lm_name] = ppl_log
    lm_types.add(f"{n_order}gram")
    print(f"build ngram {lms}!")
    return lms, ppl_results, lm_types

def build_ngram_lms(vocab: [str | VocabConfig], as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False)-> Tuple[Dict, Dict, Set[str]]:
    vocab_str = vocab if isinstance(vocab, str) else ""
    if isinstance(vocab, VocabConfig):
        assert isinstance(vocab, SentencePieceModel) and vocab.dim == 10240
        vocab_str = "spm10k"
    as_ckpt # noqa
    lms = {}
    ppl_results = {}
    lm_types = set()
    prune_threshs = [
                     None,
                     1e-9, 1.3e-8,
                     6.7e-8,
                     #3e-7, 7e-7, 1e-6, #use it if LBS
                     #1.7e-6, 4.5e-6, 1e-5, # use it if LBS
                     3e-5,
                     #7e-4, #1.7e-3, 5e-3,
                     5e-2, #8e-2,1e-1, 3e-1,5e-1, 7e-1, 9e-1,
                     ]
    prune_threshs.reverse()
    prune_threshs_sub = [5e-2, None]
    for n_order in [2,
                    #4, 5,
                    6]:
        for train_fraction in [0.1, 0.15,
                               #0.2, 0.3, #Use it if LBS
                               None,
                               ]:
            for prune_thresh in prune_threshs:
                if (n_order, prune_thresh) in [(2,None)]:
                    continue
                if n_order in [1,2] and prune_thresh is not None: # Setting for 2gram that need pruning
                    if prune_thresh < 5e-2:
                        continue
                if train_fraction == 0.15 and n_order in [5,6]:
                    continue
                if n_order > 2 and prune_thresh is not None:
                    if prune_thresh > 5e-2:
                        continue
                if train_fraction is not None and prune_thresh not in prune_threshs_sub: # Only Specific subeset of threshs are used for fraction training
                    continue
                print(f"build ngram {n_order} thresh{prune_thresh}!")
                lm, ppl_log = get_count_based_n_gram(vocab, n_order, prune_thresh, train_fraction=train_fraction, word_ppl=word_ppl,task_name=task_name, only_transcription=only_transcript)
                lm_name = f"{n_order}gram_{vocab_str}" + (f"_fr{train_fraction}".replace(".","") if train_fraction is not None else "") + f"{'_trans' if only_transcript else ''}"
                if prune_thresh:
                    lm_name += f"_pr{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
                lms[lm_name] = lm
                ppl_results[lm_name] = ppl_log
                lm_types.add(f"{n_order}gram")
    print(f"build ngram {lms}!")
    return lms, ppl_results, lm_types

def build_word_ngram_lms(word_ppl: bool = False, task_name: str = "LBS", only_transcript: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    word_ppl # noqa
    lms = {}
    ppl_results = {}
    lm_types = set()
    for n_order in [4]:
        prune_threshs = [None]
        for prune_thresh in prune_threshs:
            lm, ppl_log = get_count_based_n_gram("word", n_order, prune_thresh, task_name=task_name, only_transcription=only_transcript)
            lm_name = f"{n_order}gram_word"
            if prune_thresh:
                lm_name += f"_{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
            lms[lm_name] = lm
            ppl_results[lm_name] = ppl_log
            lm_types.add(f"{n_order}gram")
    return lms, ppl_results, lm_types

def build_apptek_ES_word_ngram_lms(word_ppl: bool = False, task_name: str = "LBS") -> Tuple[Dict, Dict, Set[str]]:
    from i6_experiments.users.zhang.experiments.language_models.n_gram import get_apptek_ES_n_gram
    word_ppl # noqa
    lms = {}
    ppl_results = {}
    lm_types = set()
    for n_order in [4]:
        prune_threshs = [None]
        for prune_thresh in prune_threshs:
            lm, ppl_log = get_apptek_ES_n_gram("word", n_order, prune_thresh, task_name=task_name)
            lm_name = f"{n_order}gram_word_2022-09-tel"
            if prune_thresh:
                lm_name += f"_{prune_thresh:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
            lms[lm_name] = lm
            ppl_results[lm_name] = ppl_log
            lm_types.add(f"{n_order}gram")
    return lms, ppl_results, lm_types

def build_ffnn_ES_lms(as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False, old: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    lms = {}
    ppl_results = {}
    lm_types = {"ffnn"}
    config = { "class": "FeedForwardLm",
                "context_size": 4,
                "dropout": 0.1,
                "embed_dim": 256,
                "embed_dropout": 0,
                "ff_hidden_dim": 1024,
                "num_layers": 2,
               }
    name_ext = "_old" if old else ('_trans' if only_transcript else '') #Default train + trans
    epochs = [50] if only_best or old else [5, 20 , 50] #[] -> last_fixed_epoch
    from i6_experiments.users.zhang.experiments.lm.ffnn import get_ES_ffnn
    for checkpoint, ppl, epoch in get_ES_ffnn(word_ppl=word_ppl, epochs=epochs, only_transcript=only_transcript, old=old):
        name = f"ffnn{config['context_size']}_{epoch}_spm10k_{task_name}{name_ext}"
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
    #print(f"ffnn:{lms}")
    return lms, ppl_results, lm_types

@lru_cache(maxsize=None)
def build_ffnn_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False, old: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    if task_name == "ES":
        return build_ffnn_ES_lms(as_ckpt=as_ckpt, word_ppl=word_ppl, only_best=only_best, task_name=task_name, only_transcript = only_transcript, old = old)
    else:
        assert task_name == "LBS", "LBS or ES"
    lms = {}
    ppl_results = {}
    lm_types = {"ffnn"}
    match = re.search(r"bpe(.+)", vocab)
    epochs = [50] if only_best else [5, 10, 20, 40, 50]
    lm_configs = {
        "std": {"context_size": 8, "num_layers": 2, "ff_hidden_dim": 2048, "dropout": 0.1},
        #"low": {"context_size": 5, "num_layers": 2, "ff_hidden_dim": 1024, "dropout": 0.2}
    }
    train_subsets = {"std": 80, "low": 30}
    for capa in lm_configs:
        config = lm_configs[capa]
        for checkpoint, ppl, epoch in get_ffnn_lm(get_vocab_by_str(vocab), **config,
                                                  word_ppl=word_ppl, epochs=epochs, train_subset=train_subsets[capa]):
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

def get_lm_by_name(lm_name:str, task_name: str = "LBS", as_ckpt: bool = True) -> Tuple[Dict, Dict, Set[str]]:
    if 'ffnn' in lm_name:
        # print(build_ffnn_lms(vocab="bpe128", as_ckpt=as_ckpt, only_best=True, task_name=task_name, only_transcript="trans" in lm_name, old="old" in lm_name)[0])
        return build_ffnn_lms(vocab="bpe128", as_ckpt=as_ckpt, only_best=False, task_name=task_name, only_transcript="trans" in lm_name, old="old" in lm_name)[0][lm_name] # for now ES LMs getter does not depend on vocab
    elif "trafo" in lm_name:
        num_layer, dim, rope_ffgated = parse_trafo_name(lm_name)
        return build_trafo_lms(vocab="bpe128", as_ckpt=as_ckpt, only_best=False, task_name=task_name, only_transcript="trans" in lm_name, old="old" in lm_name, num_layers=num_layer, dim=dim, rope_ffgated=rope_ffgated)[0][lm_name]  # for now ES LMs getter does not depend on vocab


_Lm = namedtuple("Lm", ["name", "train_version", "setup"])

_lms = {
    "n24-d512": _Lm("trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k", "v3", "lm"),
    "n96-d512": _Lm("trafo-n96-d512-gelu-drop0-b32_1k", "v3", "lm"),
    "n32-d1024": _Lm("trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k", "v3", "lm"),
    "n32-d1024-claix2023": _Lm(
        "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-spm10k", "v4", "lm_claix2023"
    ),
    "n32-d1280-claix2023": _Lm(
        "trafo-n32-d1280-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-spm10k", "v4", "lm_claix2023"
    ),
}

def build_trafo_lm_spm(as_ckpt: bool=True, word_ppl: bool = True):
    assert as_ckpt is True, "Only support ckpt return now"
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import _get_lm_model
    from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_lm_dataset, get_librispeech_lm_combined_txt
    from i6_experiments.users.zhang.datasets.vocab import GetSubwordRatioJob
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
    from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
    from i6_experiments.users.zhang.experiments.lm.lm_ppl import compute_ppl_single_epoch
    vocab = "spm10k"
    lm_dataset = get_librispeech_lm_dataset(vocab=vocab)
    #ratio = GetSubwordRatioJob(get_librispeech_lm_combined_txt(), vocab, get_returnn_subword_nmt(), apply_job=ApplySentencepieceToTextJob).out_ratio
    lms = {}
    ppl_results = {}
    lm_types = {"trafo"}
    rqmts = {"n32-d1024": {}, "n32-d1280-claix2023": {"mem":16,"gpu_mem": 24}}
    for lm_name in ["n32-d1024", "n32-d1280-claix2023"]:
        name = "trafo_spm10k" + lm_name
        lms[name] = _get_lm_model(_lms[lm_name])
        #ratio = 1
        # tk.register_output(f"LBS_{vocab}_ratio", ratio)
        ppls = compute_ppl_single_epoch(
            prefix_name=_lms[lm_name].name,
            model_with_checkpoint=lms[name],
            epoch="epoch_unk",
            dataset=lm_dataset,
            dataset_keys=["transcriptions-test-other", "transcriptions-dev-other","transcriptions-test-clean", "transcriptions-dev-clean"],
            #exponent=ratio,
            vocab=get_vocab_by_str(vocab),
            word_ppl=word_ppl,
            same_seq=True,
            batch_size=10_000,
            rqmt=rqmts[lm_name],
        )
        ppl_results[name] = ppls
    #print(ppl_results)
    return lms, ppl_results, lm_types

def build_trafo_ES_lms(as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False, old: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    lms = {}
    ppl_results = {}
    lm_types = {"trafo"}
    config = { "class": "TransformerLm",
               "num_layers": 32,
               "model_dim": 1280,
               "pos_enc": None,
               "norm": {"class": "rf.RMSNorm"},
               "ff": {"class": "returnn.frontend.decoder.transformer.FeedForwardGated"},
               "decoder_layer_opts": {
                   "self_att": {"class": "rf.RotaryPosCausalSelfAttention",
                                "with_bias": False}
               },
               "dropout": 0.0,
               "att_dropout": 0.0,}
    epochs = [100] if only_best else [40, 100] # 1, 10, 40
    epochs = [40] if old else epochs
    name_ext = "_old" if old else ('_trans' if only_transcript else '') #Default train + trans
    from i6_experiments.users.zhang.experiments.lm.trafo import get_ES_trafo
    for checkpoint, ppl, epoch in get_ES_trafo(word_ppl=word_ppl, epochs=epochs, only_transcript=only_transcript, old=old):
        name = f"trafo_{epoch}_spm10k_{task_name}{name_ext}"
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
    #print(f"trafos:{lms}")
    return lms, ppl_results, lm_types


def build_trafo_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS",
                    only_transcript: bool = False, old: bool = False, num_layers: int = 12, dim: int = 512, rope_ffgated: bool=False) -> Tuple[Dict, Dict, Set[str]]:
    if task_name == "ES":
        return build_trafo_ES_lms(as_ckpt=as_ckpt, word_ppl=word_ppl, only_best=only_best, task_name=task_name,only_transcript=only_transcript, old=old)
    else:
        assert task_name == "LBS", "LBS or ES"
    lms = {}
    ppl_results = {}
    lm_types = {"trafo"}
    match = re.search(r"bpe(.+)", vocab)
    num_layers = 12 if num_layers is None else num_layers
    dim = 512 if dim is None else dim
    config = {"num_layers": num_layers, "model_dim": dim, "dropout": 0.0, "class": "TransformerLm"}
    epochs = [50] if only_best else [20, 50] #20
    if rope_ffgated:
        epochs = [100] if only_best else [40,100]
    from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    bpe_data = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=int(match.group(1)))
    bpe = Bpe(dim=184, codes=bpe_data.bpe_codes, vocab=bpe_data.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")
    for checkpoint, ppl, epoch in get_trafo_lm(bpe, n_ep=50, bs_feat=10000, num_layers=config["num_layers"], word_ppl=word_ppl,
                                               model_dim=config["model_dim"], max_seqs=200, max_seq_length_default_target=True, epochs=epochs, rope_ffgated=rope_ffgated):
        name = f"trafo_n{num_layers}d{dim}_{epoch}_bpe{match.group(1)}{'_rope_ffgated' if rope_ffgated else ''}"
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

def build_llms(word_ppl: bool = False, task_name: str = "LBS", model_ids: List[str] = None, batch_sizes: List[int] = None,) -> Tuple[Dict, Dict, Set[str]]:
    from i6_experiments.users.zhang.experiments.lm.llm import get_llm, LLM_Batch_size
    lm_types = {"LLM"}
    model_ids = model_ids or LLM_Batch_size.keys()
    llms, ppl_llms = get_llm(model_ids=model_ids, batch_sizes=batch_sizes, task_name=task_name, word_ppl=word_ppl)
    return llms, ppl_llms, lm_types

def build_LBS_official_4gram(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False):
    from i6_experiments.users.zhang.datasets.librispeech_lm import get_4gram_binary_lm
    lm, ppls = get_4gram_binary_lm()
    return {"4gram_word_official_LBS":lm}, {"4gram_word_official_LBS":ppls}, {"4gram_official_LBS"}

def parse_ngram_name(name: str):
    """
    Parse ngram LM name strings like:
      '2gram_spm10k_fr01_pr5_0e-2'
      '6gram_spm10k_fr0.3_pr1_3e-8'
      '4gram_spm10k_pr5_3e-7'
    Returns (n_order:int, fraction:float|None, prune_thresh:float|None)
    """

    # n-order
    m = re.search(r'(\d+)gram', name)
    n_order = int(m.group(1)) if m else None

    fraction = None
    # stop before _pr or end of string
    m = re.search(r'fr([0-9.]+?)(?=_pr|$)', name)
    if m:
        raw = m.group(1)
        if '.' in raw:
            fraction = float(raw)
        else:
            if raw == '10':
                fraction = 1.0
            elif raw.startswith('0') and len(raw) > 1:
                fraction = float('0.' + raw[1:])
            else:
                fraction = float(raw)

    prune_thresh = None
    m = re.search(r'pr([0-9._eE+-]+)', name)
    if m:
        prune_str = m.group(1).replace('_', '.')
        prune_thresh = float(prune_str)


    return n_order, fraction, prune_thresh

def parse_trafo_name(name: str):
    """
    Extracts numbers after 'n' and 'd' in a string like 'trafo_n32d1280'.
    Returns a tuple (n, d) as integers.
    """
    match = re.search(r"n(\d+)d(\d+)", name)
    if not match:
        print(f"Warning: Use default trafo network setting for {name}")
        return None, None, None
    return int(match.group(1)), int(match.group(2)), "rope_ffgated" in name

def build_all_lms(vocab: [str | VocabConfig], lm_kinds: Set[str] = None, as_ckpt: bool = False, word_ppl: bool = False,
                  only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False, llmids_batch_sizes: Dict[str, int] = None) -> Tuple[Dict, Dict, Set[str]]:
    lms, ppl, types = {}, {}, set()
    if lm_kinds is None:
        lm_kinds = {"ngram", "word_ngram", "ffnn", "trafo"}
    else:
        lm_kinds = set(lm_kinds)
    builders = {
        "ngram": build_ngram_lms,
        "4gram_word_official_LBS": build_LBS_official_4gram,
        "word_ngram": build_word_ngram_lms,
        "word_ngram_apptek": build_apptek_ES_word_ngram_lms,
        "ffnn": build_ffnn_lms,
        "trafo": build_trafo_lms,
        "trafo_ES_wo_trans": functools.partial(build_trafo_lms,old=True),
        "ffnn_ES_wo_trans": functools.partial(build_ffnn_lms,old=True),
        "LLM": build_llms if llmids_batch_sizes is None
        else functools.partial(build_llms,model_ids=llmids_batch_sizes.keys(),batch_sizes=llmids_batch_sizes.values()),
    }

    for kind in lm_kinds:
        if "official" in kind:
            continue
        if kind[0].isdigit() and "gram" in kind:
            n_order, fraction, prune_thresh = parse_ngram_name(kind)
            builders.update({kind: functools.partial(build_ngram_lm,n_order=n_order, fraction=fraction, prune_thresh=prune_thresh)})
        elif kind.startswith("trafo") and "n" in kind and "d" in kind:
            num_layer, dim, rope_ffgated = parse_trafo_name(kind)
            builders.update({kind: functools.partial(build_trafo_lms, num_layers=num_layer, dim=dim, rope_ffgated=rope_ffgated)})
    # if word_ppl:
    #     bpe_ratio = None # This should be done in compute_ppl
    #     # from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_lm_combined_txt
    #     # from i6_experiments.users.zhang.datasets.vocab import GetSubwordRatioJob
    #     # from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
    #     # bpe_ratio = GetSubwordRatioJob(get_librispeech_lm_combined_txt(), vocab, get_returnn_subword_nmt()).out_ratio
    #     # tk.register_output(f"LBS_{vocab}_ratio", bpe_ratio)
    for kind, builder in builders.items():
        if kind not in lm_kinds:
            continue
        #try:
        if "ES" in kind:
            assert task_name == "ES", f"Try to get ES LM in {task_name}"
        l, p, t = builder(vocab, as_ckpt, word_ppl, only_best, task_name, only_transcript) if kind not in ["word_ngram", "word_ngram_apptek", "LLM"] else builder(word_ppl=word_ppl, task_name=task_name)
        lms.update(l)
        ppl.update(p)
        types.update(t)
        # except Exception:
        #     print(f"Something went wrong{kind}", Exception)
        #     continue

    if not as_ckpt: # Consider move this to somewhere else
        lms.update({"NoLM": None})

    return lms, ppl, types

class AggregateDictJob(Job):
    """
    reports.
    """
    def __init__(
        self,
        *,
        outputs: Dict[str, Dict[str,tk.Path | tk.Variable]],
    ):
        super(AggregateDictJob, self).__init__()
        self.outputs = outputs  # type: Dict[str, Dict[str,tk.Path | tk.Variable]]
        self.out_report_dict = self.output_path("report.py")

    def tasks(self):
        """tasks"""
        yield Task("run", rqmt={"cpu":1, "time":1})#mini_task=True)

    def run(self):
        """run"""
        import json
        res = dict()
        for name, d1 in self.outputs.items():
            avg = 0.0
            for k,v in d1.items():
                if isinstance(v,tk.Path):
                    with open(v.get_path()) as f:
                        avg += float(f.read())
                elif isinstance(v,tk.Variable):
                    avg += float(v.get())
                else:
                    raise TypeError(f"unknown type {type(v)}")
            res[name] = avg / len(d1)
        res = dict(sorted(res.items(), key=lambda item: item[1]))
        with open(self.out_report_dict.get_path(), "wt") as out:
            json.dump(res, out,indent=2)


def py():
    # from i6_experiments.users.zhang.datasets.librispeech import get_train_corpus_text, get_test_corpus_text
    # from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
    # vocab = "bpe128"
    # for key in ["dev-clean",
    #                     "dev-other",
    #                     "test-clean",
    #                     "test-other",
    #                     "train",
    #                     ]:
    #     if key == "train":
    #         getter = get_train_corpus_text()
    #     else:
    #         getter = get_test_corpus_text([key])
    #     #bpe_ratio = GetBpeRatioJob(getter, vocab, get_returnn_subword_nmt()).out_ratio
    #     #tk.register_output(f"test/LBS/bpe_ratio/{vocab}/{key}_bpe_ratio", bpe_ratio)

    from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
    from i6_experiments.users.zhang.experiments.apptek.am.ctc_spm10k_16khz_mbw import get_model_and_vocab
    from i6_experiments.users.zhang.experiments.apptek.datasets.spanish.f16kHz.data import DEV_KEYS, TEST_KEYS
    EXCLUDE_LIST = ["napoli", "callcenter", "voice_call", "tvshows", "mtp_eval-v2"]
    EVAL_DATASET_KEYS = [f"{key}" for key in TEST_KEYS if
                         not any(exclude in key for exclude in EXCLUDE_LIST)]
    _, spm, _ = get_model_and_vocab(fine_tuned_model=True)

    # for k, v in spm["vocabulary"].items():
    #     print(f"{k}: {v}")
    # print(f"vocab setting: {spm}")
    vocab_config = SentencePieceModel(dim=spm["vocabulary"]["vocabulary_size"], model_file=spm["spm"])
    # Define search space
    # n_orders = [3, 4, 5, 6]  # adjust as you like
    # prune_thresholds = [1e-9, 3e-8, 1e-7, 3e-7]  # smooth progression
    # fractions = [1.0, 0.8, 0.6, 0.4, 0.3]  # decreasing training data

    # n_orders = [6, 5]  # 6-gram first, then maybe 5-gram
    # fractions = [1.0, 0.8, 0.6]
    # prune_thresholds = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]

    n_orders = [2,3]#[6, 5, 3]
    fractions = [1.0, 0.6, 0.3, 0.1, 0.03]
    prune_thresholds = [1e-3]#, 3e-3, 1e-2, 3e-2, 1e-1]

    # Collect results
    ppl_dict = {}

    for n in n_orders:
        for pr in prune_thresholds:
            for fr in fractions:
                print(f"\n=== Building {n}-gram | fraction={fr} | prune={pr:.1e} ===")
                try:
                    _, ppl_results, _ = build_ngram_lm(
                        vocab=vocab_config,
                        n_order=n,
                        prune_thresh=pr,
                        fraction=fr if fr<1 else None,
                        word_ppl=False,
                        only_transcript=False,
                        task_name="ES",
                        eval_keys=set(EVAL_DATASET_KEYS),
                    )
                    # merge the returned dictionary (lm_name → ppl_log)
                    ppl_dict.update(ppl_results)

                except Exception as e:
                    print(f"⚠️ Failed for n={n}, prune={pr}, frac={fr}: {e}")

    # Done
    print("\nCollected PPL results:")
    #from i6_experiments.users.zhang.utils.report import ReportDictJob
    tk.register_output(f"test/ES_ngram_PPL_range/report_n{len(n_orders)}_p{len(prune_thresholds)}_f{len(fractions)}", AggregateDictJob(outputs=ppl_dict).out_report_dict)
    # for name, ppl in ppl_dict.items():
    #     print(f"{name}: {ppl:.3f}")