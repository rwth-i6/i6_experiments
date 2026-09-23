"""
CTC + word-level n-gram decoding with KenLM, for MLS-German.

Why: the published MLS limited-supervision numbers (wav2vec 2.0 / XLSR, arXiv 2006.13979 Tab. 5) are CTC +
KenLM n-gram decodes. Our German arms had no LM decode at all. This decodes our model's CTC output with the
OFFICIAL MLS German 5-gram (the MLS LM release, trained on exactly the ``data.txt`` our text injection uses;
the MLS paper's own baseline LM), converted losslessly to a KenLM trie binary.

Decoder: pyctcdecode (BPE-aware CTC beam search + KenLM word scoring), NOT flashlight's lexicon decoder:
with word-piece spellings (no separate word-boundary token) flashlight scores every frame after a word end
with the silence token's emission (LexiconDecoder.cpp, "Try same lexicon node": ``n = root ? sil : prevIdx``),
so a CTC repeat of a word's last piece is priced as silence and the search derails (measured 2026-09-23:
"der hund und die katze" with 2x repeated frames -> "derbe hundes und dien katze").

Normalisation: the decoder emits lowercase words in the model's spelling (sharp-s -> ss, hyphen-split; the ONE
German normalisation, ``german_xling.normalize_german_text``); :class:`NormMappedLanguageModel` scores each as
the LM's most frequent word with the same normalised form, and treats it as unknown only if there is none.

Pipeline:

1. :func:`ctc_topk_forward` (GPU, via ``forward_to_hdf``): the arm's CTC log-probs, top-K per frame
   (full 10,244-wide log-probs would be ~35 GB per eval set; CTC is peaky, K=64 keeps all mass that matters).
2. :class:`CtcKenLmDecodeJob` (CPU, login-node LocalEngine): alpha/beta tuned on a fixed dev subset,
   full dev + test decoded with the best pair. pyctcdecode + kenlm are not in the shared py env: they live in
   a private target dir and load via ``kenlm_env.sh`` (``clusters/fzj.md``), so the decode runs as a subprocess.
3. Scoring: the task's own ``score_recog_output_func`` (the German sclite chain), like every German number.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk

KENLM_ENV = "/e/project1/spell/koch13/pylibs/kenlm_env.sh"
MLS_DE_LM_DIR = "/e/project1/spell/koch13/mls_lm/mls_lm_german"


def ctc_topk_forward(source, *, in_spatial_dim, model):
    """
    RETURNN forward def: top-K CTC log-probs of layer ``ctc_dump_layer`` (config), no soft collapse,
    no prior. Outputs "output" (log-probs, [B, T, K]) and "topk_idx" (int32, [B, T, K]).
    """
    import returnn.frontend as rf
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict

    config = get_global_config()
    layer = config.int("ctc_dump_layer", 0)
    assert layer > 0, "ctc_dump_layer must be set"
    expected = rf.get_run_ctx().expected_outputs["output"]
    k_dim = expected.dims[-1]

    if source.feature_dim and source.feature_dim.dimension == 1:
        source = rf.squeeze(source, axis=source.feature_dim)
    collected = CollectOutputsDict(allowed_key_patterns=[str(layer - 1)])
    _, enc_spatial_dim = model.encode_no_transform(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected)
    logits = getattr(model, f"enc_aux_logits_{layer}")(collected[str(layer - 1)])
    log_probs = rf.log_softmax(logits, axis=model.wb_target_dim)
    values, indices, _ = rf.top_k(log_probs, axis=model.wb_target_dim, k_dim=k_dim)
    values = rf.cast(values, "float32")
    indices = rf.cast(indices, "int32")
    indices.sparse_dim = None  # plain int32 ids (the declared output has no sparse dim)
    # the config's model_outputs dims are templates: bind the time dim to the encoder frames
    # (as zeyer's gauss_hmm forward does; replace_dim leaves the template's sizes unset for the callback)
    for key in ("output", "topk_idx"):
        rf.get_run_ctx().expected_outputs[key].dims[1].declare_same_as(enc_spatial_dim)
    dims = [*values.remaining_dims((enc_spatial_dim, k_dim)), enc_spatial_dim, k_dim]
    rf.get_run_ctx().mark_as_output(values, "output", dims=dims)
    rf.get_run_ctx().mark_as_output(indices, "topk_idx", dims=dims)


def get_ctc_topk_hdf(*, dataset, model, extra_config: Optional[Dict[str, Any]], ctc_layer: int, k: int = 64) -> tk.Path:
    """``forward_to_hdf`` of :func:`ctc_topk_forward` over ``dataset`` (a DatasetConfig)."""
    from returnn.tensor import Dim, batch_dim
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    time_dim = Dim(None, name="ctc_frames")
    k_dim = Dim(k, name="ctc_topk")
    config = {
        **(extra_config or {}),
        "ctc_dump_layer": ctc_layer,
        "model_outputs": {
            # feature_dim must be explicit, else the HDF writer sees dim=None and flattens [T, K] to [T]
            "output": {"dims": [batch_dim, time_dim, k_dim], "dtype": "float32", "feature_dim": k_dim},
            "topk_idx": {"dims": [batch_dim, time_dim, k_dim], "dtype": "int32", "feature_dim": k_dim},
        },
    }
    return forward_to_hdf(dataset=dataset, model=model, forward_def=ctc_topk_forward, config=config)


class CtcKenLmDecodeJob(Job):
    """
    KenLM CTC beam search (pyctcdecode) on dumped top-K CTC log-probs. Tunes ``lm_weight`` (pyctcdecode alpha)
    x ``word_score`` (beta) on a fixed dev subset (WER against ``dev_text_dict``,
    same normalisation as the references), then decodes the full dev and test sets with the best pair.
    Outputs py-dict search results {seq_tag: text} for the task's scorer, plus the tuning grid.
    """

    def __init__(
        self,
        *,
        dev_hdf: tk.Path,
        test_hdf: tk.Path,
        dev_text_dict: tk.Path,
        spm_model: tk.Path,
        lm_binary: tk.Path,
        lm_vocab: tk.Path,
        lm_weights: Sequence[float] = (0.3, 0.5, 0.7, 1.0),
        word_scores: Sequence[float] = (0.0, 1.0, 2.0),
        beam_size: int = 100,
        tune_num_seqs: int = 1000,
        num_procs: int = 14,
        version: int = 1,
    ):
        self.dev_hdf = dev_hdf
        self.test_hdf = test_hdf
        self.dev_text_dict = dev_text_dict
        self.spm_model = spm_model
        self.lm_binary = lm_binary
        self.lm_vocab = lm_vocab
        self.lm_weights = list(lm_weights)
        self.word_scores = list(word_scores)
        self.beam_size = beam_size
        self.tune_num_seqs = tune_num_seqs
        self.num_procs = num_procs
        self.version = version
        self.out_dev = self.output_path("search_dev.py.gz")
        self.out_test = self.output_path("search_test.py.gz")
        self.out_grid = self.output_path("tuning_grid.json")
        self.out_lexicon = self.output_path("norm_to_lm_spelling.txt")
        # LocalEngine (login node, cpus=40): stay well below it (clusters/fzj.md, the fill-the-engine flood)
        self.rqmt = {"cpu": num_procs, "mem": 100, "time": 12}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        args = {
            "dev_hdf": self.dev_hdf.get_path(),
            "test_hdf": self.test_hdf.get_path(),
            "dev_text_dict": self.dev_text_dict.get_path(),
            "spm_model": self.spm_model.get_path(),
            "lm_binary": self.lm_binary.get_path(),
            "lm_vocab": self.lm_vocab.get_path(),
            "lm_weights": self.lm_weights,
            "word_scores": self.word_scores,
            "beam_size": self.beam_size,
            "tune_num_seqs": self.tune_num_seqs,
            "num_procs": self.num_procs,
            "out_dev": self.out_dev.get_path(),
            "out_test": self.out_test.get_path(),
            "out_grid": self.out_grid.get_path(),
            "out_lexicon": self.out_lexicon.get_path(),
        }
        with open("decode_args.json", "w") as f:
            json.dump(args, f, indent=1)
        # the worker's sys.path (recipe/, sisyphus) must reach the subprocess: this module imports sisyphus
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        subprocess.check_call([KENLM_ENV, sys.executable, os.path.abspath(__file__), "decode_args.json"], env=env)


# ---------------------------------------------------------------------------------------------------------
# Standalone decoding (run as a subprocess with kenlm_env.sh; no sisyphus imports needed below)

_DEC = None
_ARGS = None
_NORM2LM = None


def _norm(line: str) -> str:
    """Keep in sync with german_xling.normalize_german_text (upper, sharp-s -> SS via str.upper, hyphen split)."""
    return line.upper().replace("-", " ")


def _load_hdf(path):
    import h5py
    import numpy

    with h5py.File(path, "r") as f:
        tags = [t.decode("utf8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][()]]
        vals = f["inputs"][()]  # main key "output" -> stored as "inputs", flattened [sum_T, K]
        lens = f["seqLengths"][()][:, 0]
        idx = f["targets/data/topk_idx"][()]
    offs = numpy.concatenate([[0], numpy.cumsum(lens)])
    return tags, [(vals[offs[i] : offs[i + 1]], idx[offs[i] : offs[i + 1]]) for i in range(len(tags))]


def _labels(sp):
    """pyctcdecode labels in CTC output order: SPM pieces (lowercased, BPE-style "▁"), blank "" last."""
    labels = []
    for i in range(sp.get_piece_size()):
        p = sp.id_to_piece(i)
        labels.append(f"▁<{i}>" if p.startswith("<") and p.endswith(">") else p.lower())
    return labels + [""]


def _make_lm_class():
    from pyctcdecode.language_model import LanguageModel, _get_empty_lm_state, LOG_BASE_CHANGE_FACTOR

    class NormMappedLanguageModel(LanguageModel):
        """
        The official MLS 5-gram, seen through our normalisation: the decoder produces lowercase words in the
        model's spelling (sharp-s -> ss, no hyphens); each is scored as the LM's most frequent word with the
        same normalised form (e.g. "strasse" -> "straße"), and counts as unknown only if no LM word has that form.
        """

        def __init__(self, kenlm_model, norm2lm, **kwargs):
            import pygtrie

            super().__init__(kenlm_model, unigrams=None, **kwargs)
            self._norm2lm = norm2lm
            self._unigram_set = set(norm2lm)
            self._char_trie = pygtrie.CharTrie.fromkeys(self._unigram_set)

        def score(self, prev_state, word, is_last_word=False):
            end_state = _get_empty_lm_state()
            lm_word = self._norm2lm.get(word)
            if lm_word is None:
                lm_score = self._kenlm_model.BaseScore(prev_state, word, end_state) + self.unk_score_offset
            else:
                lm_score = self._kenlm_model.BaseScore(prev_state, lm_word, end_state)
            if is_last_word:
                lm_score = lm_score + self._get_raw_end_score(end_state)
            lm_score = self.alpha * lm_score * LOG_BASE_CHANGE_FACTOR + self.beta
            return lm_score, end_state

    return NormMappedLanguageModel


def _init_worker(a, cfg):
    global _DEC, _ARGS
    import kenlm
    import sentencepiece
    from pyctcdecode.alphabet import Alphabet
    from pyctcdecode.decoder import BeamSearchDecoderCTC

    _ARGS = a
    sp = sentencepiece.SentencePieceProcessor(model_file=a["spm_model"])
    lm_cls = _make_lm_class()
    lm = lm_cls(kenlm.Model(a["lm_binary"]), _NORM2LM, alpha=cfg[0], beta=cfg[1])
    _DEC = BeamSearchDecoderCTC(Alphabet.build_alphabet(_labels(sp)), language_model=lm)


def _decode_one(item):
    import numpy

    tag, vals, idx = item
    vocab = _ARGS["_vocab"]
    em = numpy.full((vals.shape[0], vocab), -100.0, dtype=numpy.float32)
    numpy.put_along_axis(em, idx.astype(numpy.int64), vals, axis=1)
    text = _DEC.decode(em, beam_width=_ARGS["beam_size"])
    return tag, _norm(text)


def _decode_set(a, cfg, items):
    import multiprocessing

    with multiprocessing.get_context("fork").Pool(a["num_procs"], initializer=_init_worker, initargs=(a, cfg)) as pool:
        return dict(pool.imap_unordered(_decode_one, items, chunksize=4))


def _wer(hyps: Dict[str, str], refs: Dict[str, str]) -> float:
    import torchaudio

    errs = words = 0
    for tag, hyp in hyps.items():
        r = refs[tag].split()
        errs += torchaudio.functional.edit_distance(r, hyp.split())
        words += len(r)
    return 100.0 * errs / max(words, 1)


def _write_py_dict(path, d: Dict[str, str]):
    import gzip

    with gzip.open(path, "wt", encoding="utf8") as f:
        f.write("{\n")
        for tag in sorted(d):
            f.write(f"{tag!r}: {d[tag]!r},\n")
        f.write("}\n")


def main(args_file: str):
    import ast
    import gzip
    import random
    import sentencepiece

    a = json.load(open(args_file))
    global _NORM2LM
    sp = sentencepiece.SentencePieceProcessor(model_file=a["spm_model"])
    a["_vocab"] = sp.get_piece_size() + 1  # + blank

    # normalised form -> the LM's most frequent spelling of it (vocab_counts.txt is sorted by count)
    norm2lm = {}
    n_in = 0
    with open(a["lm_vocab"], encoding="utf8") as fin:
        for line in fin:
            parts = line.split()
            if not parts:
                continue
            n_in += 1
            form = _norm(parts[0]).split()
            if len(form) != 1:
                continue  # hyphenated: our normalisation splits it into several words
            norm2lm.setdefault(form[0].lower(), parts[0])
    _NORM2LM = norm2lm
    with open(a["out_lexicon"], "w", encoding="utf8") as fout:
        for k, v in norm2lm.items():
            if k != v:
                fout.write(f"{k} {v}\n")  # only the entries where the spelling differs, for inspection
    n_out = len(norm2lm)
    print(f"LM words: {n_in:,}; normalised forms: {n_out:,}", flush=True)

    refs_raw = ast.literal_eval(
        gzip.open(a["dev_text_dict"], "rt").read()
        if a["dev_text_dict"].endswith(".gz")
        else open(a["dev_text_dict"]).read()
    )
    refs = {k: _norm(v) for k, v in refs_raw.items()}

    dev_tags, dev_data = _load_hdf(a["dev_hdf"])
    dev_items = [(t, v, i) for t, (v, i) in zip(dev_tags, dev_data)]
    rng = random.Random(42)
    tune_items = rng.sample(dev_items, min(a["tune_num_seqs"], len(dev_items)))

    grid = {}
    for lw in a["lm_weights"]:
        for ws in a["word_scores"]:
            hyps = _decode_set(a, (lw, ws), tune_items)
            grid[f"{lw},{ws}"] = w = _wer(hyps, refs)
            print(f"tune lm_weight={lw} word_score={ws}: WER {w:.2f} on {len(tune_items)} dev seqs", flush=True)
    best = min(grid, key=grid.get)
    lw, ws = map(float, best.split(","))
    print(f"best: lm_weight={lw} word_score={ws}", flush=True)

    dev_hyps = _decode_set(a, (lw, ws), dev_items)
    test_tags, test_data = _load_hdf(a["test_hdf"])
    test_hyps = _decode_set(a, (lw, ws), [(t, v, i) for t, (v, i) in zip(test_tags, test_data)])
    _write_py_dict(a["out_dev"], dev_hyps)
    _write_py_dict(a["out_test"], test_hyps)
    json.dump(
        {
            "grid": grid,
            "best": {"lm_weight": lw, "word_score": ws},
            "dev_wer_internal": _wer(dev_hyps, refs),
            "lexicon_words": n_out,
            "tune_num_seqs": len(tune_items),
        },
        open(a["out_grid"], "w"),
        indent=1,
    )


if __name__ == "__main__":
    main(sys.argv[1])
