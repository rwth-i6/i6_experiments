from typing import Tuple, Dict, Set, List, Optional, Union, Type

from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
from returnn_common.datasets_old_2022_10.interface import VocabConfig

from .language_models.n_gram import get_count_based_n_gram, get_apptek_ES_n_gram
from .lm.ffnn import get_ffnn_lm
from .lm.trafo import get_trafo_lm
import re
from sisyphus import Job, Task, tk, gs

from functools import lru_cache
from collections import namedtuple


def build_ngram_lms(vocab: [str | VocabConfig], as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False)-> Tuple[Dict, Dict, Set[str]]:
    print(f"start build ngram!")
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

def build_ffnn_ES_lms(as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False) -> Tuple[Dict, Dict, Set[str]]:
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
    epochs = [50] if only_best else [1, 10 , 20, 50] #[] -> last_fixed_epoch
    from i6_experiments.users.zhang.experiments.lm.ffnn import get_ES_ffnn
    for checkpoint, ppl, epoch in get_ES_ffnn(word_ppl=word_ppl, epochs=epochs, only_transcript=only_transcript):
        name = f"ffnn{config['context_size']}_{epoch}_spm10k_{task_name}{'_trans' if only_transcript else ''}"
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
def build_ffnn_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    if task_name == "ES":
        return build_ffnn_ES_lms(as_ckpt=as_ckpt, word_ppl=word_ppl, only_best=only_best, task_name=task_name, only_transcript = only_transcript)
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
        return build_ffnn_lms(vocab="bpe128", as_ckpt=as_ckpt, only_best=True, task_name=task_name, only_transcript=True if "trans" in lm_name else False)[0][lm_name] # for now ES LMs getter does not depend on vocab
    elif "trafo" in lm_name:
        return build_trafo_lms(vocab="bpe128", as_ckpt=as_ckpt, only_best=True, task_name=task_name, only_transcript=True if "trans" in lm_name else False)[0][lm_name]  # for now ES LMs getter does not depend on vocab


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
    rqmts = {"n32-d1024": {}, "n32-d1280-claix2023": {"gpu_mem": 24}}
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
            dataset_keys=["transcriptions-test-other", "transcriptions-dev-other"],
            #exponent=ratio,
            word_ppl=word_ppl,
            same_seq=True,
            batch_size=10_000,
            rqmt=rqmts[lm_name],
        )
        ppl_results[name] = ppls
    #print(ppl_results)
    return lms, ppl_results, lm_types

def build_trafo_ES_lms(as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False) -> Tuple[Dict, Dict, Set[str]]:
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
    epochs = [40] if only_best else [10, 40] # 1, 10, 40
    from i6_experiments.users.zhang.experiments.lm.trafo import get_ES_trafo
    for checkpoint, ppl, epoch in get_ES_trafo(word_ppl=word_ppl, epochs=epochs, only_transcript=only_transcript):
        name = f"trafo_{epoch}_spm10k_{task_name}{'_trans' if only_transcript else ''}"
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

def build_trafo_lms(vocab: str, as_ckpt: bool=False, word_ppl: bool = False, only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    if task_name == "ES":
        return build_trafo_ES_lms(as_ckpt=as_ckpt, word_ppl=word_ppl, only_best=only_best, task_name=task_name,only_transcript=only_transcript)
    else:
        assert task_name == "LBS", "LBS or ES"
    lms = {}
    ppl_results = {}
    lm_types = {"trafo"}
    match = re.search(r"bpe(.+)", vocab)
    config = {"num_layers": 12, "model_dim": 512, "dropout": 0.0, "class": "TransformerLm"}
    epochs = [50] if only_best else [20, 50] #20
    from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe
    from i6_experiments.users.zeyer.datasets.utils.bpe import Bpe
    bpe_data = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=int(match.group(1)))
    bpe = Bpe(dim=184, codes=bpe_data.bpe_codes, vocab=bpe_data.bpe_vocab, eos_idx=0, bos_idx=0, unknown_label="<unk>")
    for checkpoint, ppl, epoch in get_trafo_lm(bpe, n_ep=50, bs_feat=10000, num_layers=config["num_layers"], word_ppl=word_ppl,
                                               model_dim=config["model_dim"], max_seqs=200, max_seq_length_default_target=True, epochs=epochs):
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

def build_llms(word_ppl: bool = False, task_name: str = "LBS") -> Tuple[Dict, Dict, Set[str]]:
    from i6_experiments.users.zhang.experiments.lm.llm import get_llm, LLM_Batch_size
    lm_types = {"LLM"}
    model_ids = LLM_Batch_size.keys()
    llms, ppl_llms = get_llm(model_ids=model_ids, batch_sizes=[LLM_Batch_size[key] for key in model_ids], task_name=task_name, word_ppl=word_ppl)
    return llms, ppl_llms, lm_types

def build_all_lms(vocab: [str | VocabConfig], lm_kinds: Set[str] = None, as_ckpt: bool = False, word_ppl: bool = False,
                  only_best: bool = False, task_name: str = "LBS", only_transcript: bool = False) -> Tuple[Dict, Dict, Set[str]]:
    lms, ppl, types = {}, {}, set()
    if lm_kinds is None:
        lm_kinds = {"ngram", "word_ngram", "ffnn", "trafo"}
    else:
        lm_kinds = set(lm_kinds)
    builders = {
        "ngram": build_ngram_lms,
        "word_ngram": build_word_ngram_lms,
        "word_ngram_apptek": build_apptek_ES_word_ngram_lms,
        "ffnn": build_ffnn_lms,
        "trafo": build_trafo_lms,
        "LLM": build_llms,
    }
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

def py():
    from i6_experiments.users.zhang.datasets.librispeech import get_train_corpus_text, get_test_corpus_text
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
    vocab = "bpe128"
    for key in ["dev-clean",
                        "dev-other",
                        "test-clean",
                        "test-other",
                        "train",
                        ]:
        if key == "train":
            getter = get_train_corpus_text()
        else:
            getter = get_test_corpus_text([key])
        #bpe_ratio = GetBpeRatioJob(getter, vocab, get_returnn_subword_nmt()).out_ratio
        #tk.register_output(f"test/LBS/bpe_ratio/{vocab}/{key}_bpe_ratio", bpe_ratio)