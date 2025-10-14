"""
Use a prior to rescore some recog output.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sisyphus import Job, Task, tk
from i6_core import util
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from .rescoring import combine_scores

import torch
import gzip
import numpy as np
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

@dataclass
class Prior:
    """Prior"""

    file: tk.Path  # will be read via numpy.loadtxt
    type: str  # "log_prob" or "prob"
    vocab: tk.Path  # line-based, potentially gzipped


def prior_score(res: RecogOutput, *, prior: Prior) -> RecogOutput:
    """
    Use prior to score some recog output.

    :param res: previous recog output, some hyps to rescore. the score in those hyps is ignored
    :param prior:
    :return: recog output with prior scores instead
    """
    return RecogOutput(
        output=SearchPriorRescoreJob(
            res.output, prior=prior.file, prior_type=prior.type, vocab=prior.vocab
        ).out_search_results
    )


def prior_rescore(res: RecogOutput, *, prior: Prior, prior_scale: float, orig_scale: float = 1.0) -> RecogOutput:
    """
    Use prior to rescore some recog output, i.e. combine the orig score with new prior score.

    :param res: previous recog output, some hyps to rescore. the score in those hyps is ignored
    :param prior:
    :param prior_scale: scale for the prior. the negative of this will be used for the weight
    :param orig_scale: scale for the original score
    :return: recog output with combined scores
    """
    scores = [(orig_scale, res), (-prior_scale, prior_score(res, prior=prior))]
    return combine_scores(scores)

def prior_rescore_force_align_def(
    *,
    model,
    data,
    data_spatial_dim,
    targets,
    targets_beam_dim,
    targets_spatial_dim,
    **_other,
) -> Tensor:
    """RescoreDef API: CTC forced-align + prior rescoring.

    - Uses torchaudio.functional.forced_align on the model's per-frame log-probs.
    - Sums log-prior over aligned tokens (ignoring blanks and frames beyond input length).
    - Returns a RETURNN Tensor with dims (batch, targets_beam_dim).
    """
    """RescoreDef API: CTC forced-align + frame-wise prior (includes blank)."""
    import gzip
    import numpy as np
    import torch
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim
    from torchaudio.functional import forced_align  # batch size must be 1 at the moment

    # ---------- helpers ----------
    def _path_str(p):
        return p.get_path() if hasattr(p, "get_path") else str(p)

    def _read_lines(path: str):
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return [ln.rstrip("\n") for ln in f]
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            return [ln.rstrip("\n") for ln in f]

    def _numpy_loadtxt_path(p):
        return np.loadtxt(_path_str(p))

    def _remap_prior_to_model_vocab(prior_log_vec: np.ndarray, prior_vocab, model_vocab, V_expected: int) -> np.ndarray:
        if model_vocab is None:
            if prior_log_vec.shape[0] != V_expected:
                raise ValueError(f"prior length {prior_log_vec.shape[0]} != model V {V_expected} and no model_vocab to remap")
            return prior_log_vec
        if len(model_vocab) != V_expected:
            raise ValueError(f"model_vocab length {len(model_vocab)} != V {V_expected}")
        prior_index = {tok: i for i, tok in enumerate(prior_vocab)}
        remapped = np.full((V_expected,), fill_value=-1e30, dtype=np.float32)
        for j, tok in enumerate(model_vocab):
            i = prior_index.get(tok)
            if i is not None:
                remapped[j] = prior_log_vec[i]
        return remapped

    # ---------- model forward ----------
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
    assert logits.feature_dim is not None and enc_spatial_dim in logits.dims

    log_probs = model.log_probs_wb_from_logits(logits)  # RETURNN Tensor
    emissions: torch.Tensor = log_probs.raw_tensor       # (B, T, V), **log-softmaxed**
    assert isinstance(emissions, torch.Tensor) and emissions.dim() == 3
    B, T, V = emissions.shape
    device = emissions.device
    dtype = emissions.dtype

    # encoder valid lengths per sample
    input_lengths_rt: Tensor = enc_spatial_dim.dyn_size_ext
    input_lengths: torch.Tensor = input_lengths_rt.raw_tensor  # (B,)
    assert input_lengths.shape == (B,)

    # ---------- load prior (frame-wise, including blank) ----------
    prior_obj = _other.get("prior", None)
    if prior_obj is None:
        raise ValueError("Expected `_other['prior']` (Prior dataclass with fields file/type/vocab).")

    prior_vec = _numpy_loadtxt_path(prior_obj.file).astype(np.float32)  # (|prior_vocab|,)
    ptype = getattr(prior_obj, "type", "log_prob")
    if ptype == "log_prob":
        prior_log_vec = prior_vec
    elif ptype == "prob":
        prior_log_vec = np.log(np.maximum(prior_vec, np.finfo(np.float32).tiny))
    else:
        raise ValueError(f"invalid prior.type {ptype!r}; expected 'log_prob' or 'prob'")

    prior_vocab = _read_lines(_path_str(prior_obj.vocab))
    if len(prior_vocab) != prior_log_vec.shape[0]:
        raise ValueError(f"prior vocab size {len(prior_vocab)} != prior vector length {prior_log_vec.shape[0]}")

    model_vocab = _other.get("model_vocab", None)  # optional List[str] of length V
    prior_log_vec = _remap_prior_to_model_vocab(prior_log_vec, prior_vocab, model_vocab, V)
    prior_log_t = torch.from_numpy(prior_log_vec).to(device=device, dtype=dtype)

    blank_idx = int(model.blank_idx)

    # ---------- set up dims for return ----------
    rem_dims = targets.remaining_dims(targets_spatial_dim)  # {batch_dim, targets_beam_dim}
    batch_dim = next(d for d in rem_dims if d is not targets_beam_dim)
    assert isinstance(batch_dim, Dim)

    # ---------- scoring: loop over beam, then over batch (forced_align requires B=1) ----------
    per_beam_scores = []
    for beam_idx in range(targets_beam_dim.get_dim_value()):
        # slice per-beam targets
        #import pdb;pdb.set_trace()
        targets_b = rf.gather(targets, axis=targets_beam_dim, indices=beam_idx)
        targets_b_seq_lens = rf.gather(targets_spatial_dim.dyn_size_ext, axis=targets_beam_dim, indices=beam_idx)
        targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{targets_spatial_dim.name}_beam{beam_idx}")
        targets_b, _ = rf.replace_dim(targets_b, in_dim=targets_spatial_dim, out_dim=targets_b_spatial_dim)
        targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)

        tokens_full: torch.Tensor = targets_b.raw_tensor                 # (B, Lc_max)
        tokens_lens: torch.Tensor = targets_b_spatial_dim.dyn_size_ext.raw_tensor  # (B,)
        tokens_full = tokens_full.to(device=device, dtype=torch.long)
        tokens_lens = tokens_lens.to(device=device, dtype=torch.long)

        beam_scores_list = []
        for b in range(B):
            T_b = int(input_lengths[b].item())
            L_b = int(tokens_lens[b].item())
            if T_b <= 0 or L_b <= 0:
                beam_scores_list.append(torch.tensor(0.0, device=device, dtype=dtype))
                continue

            # Slice per-sample emissions and tokens
            emis_b = emissions[b, :T_b, :]                            # (T_b, V)
            toks_b = tokens_full[b, :L_b]                             # (L_b,)

            with torch.no_grad():
                align_b, _ = forced_align(
                    log_probs=emis_b.unsqueeze(0),           # (1, T_b, V)
                    targets=toks_b.unsqueeze(0),           # (1, L_b)
                    input_lengths=torch.tensor([T_b], device=device),
                    target_lengths=torch.tensor([L_b], device=device),
                    blank=blank_idx,
                )  # -> (1, T_b), long

            align_b = align_b.squeeze(0)           # (T_b,)
            # Frame-wise prior sum INCLUDING blanks
            # score_b = sum_t log prior[ align_b[t] ]
            score_b = prior_log_t[align_b].sum()   # scalar tensor
            beam_scores_list.append(score_b)

        # stack batch back: (B,)
        per_beam_scores.append(torch.stack(beam_scores_list, dim=0))

    # stack beams: (B, beam)
    scores_bt = torch.stack(per_beam_scores, dim=1)

    # ---------- wrap as RETURNN Tensor with dims (batch, beam) ----------
    scores = Tensor(
        "scores",
        dims=[batch_dim, targets_beam_dim],
        dtype="float32",
        raw_tensor=scores_bt,
    )
    return scores


class SearchPriorRescoreJob(Job):
    """
    Use prior to rescore some recog output.
    """

    __sis_version__ = 2

    def __init__(
        self, search_py_output: tk.Path, *, prior: tk.Path, prior_type: str, vocab: tk.Path, output_gzip: bool = True
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param prior:
        :param prior_type: "log_prob" or "prob"
        :param vocab: line-based, potentially gzipped
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        self.prior = prior
        assert prior_type in ["log_prob", "prob"], f"invalid prior_type {prior_type!r}"
        self.prior_type = prior_type
        self.vocab = vocab
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", rqmt = {"time": 1, "cpu": 1, "mem": 20})

    def run(self):
        """run"""
        import numpy as np
        #import tracemalloc

       # tracemalloc.start()
        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string
        #current, peak = tracemalloc.get_traced_memory()
        #print(f"Current: {current / 1024 ** 2:.1f} MB; Peak: {peak / 1024 ** 2:.1f} MB")

        #tracemalloc.stop()

        vocab: List[str] = util.uopen(self.vocab, "rt").read().splitlines()
        vocab_to_idx: Dict[str, int] = {word: i for (i, word) in enumerate(vocab)}

        prior = np.loadtxt(self.prior.get_path())
        assert prior.shape == (len(vocab),), f"prior shape {prior.shape} vs vocab size {len(vocab)}"
        # The `type` is about what is stored in the file.
        # We always want it in log prob here, so we potentially need to convert it.
        if self.prior_type == "log_prob":
            pass  # already log prob
        elif self.prior_type == "prob":
            prior = np.log(prior)
        else:
            raise ValueError(f"invalid static_prior type {self.prior_type!r}")

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]. we ignore the input score.
                out.write(f"{seq_tag!r}: [\n")
                for _, text in entry:
                    assert isinstance(text, str)
                    scores = []
                    for label in text.split():
                        if label not in vocab_to_idx:
                            raise ValueError(f"unknown label {label!r} in seq_tag {seq_tag!r}, seq {text!r}")
                        scores.append(prior[vocab_to_idx[label]])
                    score = float(np.sum(scores))
                    out.write(f"({score}, {text!r}),\n")
                out.write("],\n")
            out.write("}\n")


class PriorRemoveLabelRenormJob(Job):
    """
    Gets some prior, removes some label from it, renorms the remaining.
    """

    def __init__(self, *, prior_file: tk.Path, prior_type: str, vocab: tk.Path, remove_label: str, out_prior_type: str):
        self.prior_file = prior_file
        self.prior_type = prior_type
        self.vocab = vocab
        self.remove_label = remove_label
        self.out_prior_type = out_prior_type

        self.out_prior = self.output_path("prior.txt")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import numpy as np

        vocab: List[str] = util.uopen(self.vocab, "rt").read().splitlines()
        vocab_to_idx: Dict[str, int] = {word: i for (i, word) in enumerate(vocab)}

        assert (
            vocab.count(self.remove_label) == 1
        ), f"remove_label {self.remove_label!r} not unique in vocab. found {vocab.count(self.remove_label)} times."
        remove_label_idx = vocab_to_idx[self.remove_label]

        prior = np.loadtxt(self.prior_file.get_path())
        assert prior.shape == (len(vocab),), f"prior shape {prior.shape} vs vocab size {len(vocab)}"
        # The `type` is about what is stored in the file.
        # We always want it in log prob here, so we potentially need to convert it.
        if self.prior_type == "log_prob":
            pass  # already log prob
        elif self.prior_type == "prob":
            prior = np.log(prior)
        else:
            raise ValueError(f"invalid static_prior type {self.prior_type!r}")

        neg_inf = float("-inf")

        def _logsumexp(arg: np.ndarray) -> np.ndarray:
            """
            Stable log sum exp.
            """
            if np.all(arg == neg_inf):
                return arg
            a_max = np.max(arg)
            lsp = np.log(np.sum(np.exp(arg - a_max)))
            return a_max + lsp

        prior = np.concatenate([prior[:remove_label_idx], prior[remove_label_idx + 1 :]])
        assert prior.shape == (len(vocab) - 1,), f"prior shape {prior.shape} vs vocab size {len(vocab) - 1}"
        prior = prior - _logsumexp(prior)

        if self.out_prior_type == "log_prob":
            pass
        elif self.out_prior_type == "prob":
            prior = np.exp(prior)
        else:
            raise ValueError(f"invalid out_prior_type {self.out_prior_type!r}")
        np.savetxt(self.out_prior.get_path(), prior)
