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
from torchaudio.functional import forced_align
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


def _read_lines(path: str) -> List[str]:
    """Read a line-based vocab file; supports plain text or .gz."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return [ln.rstrip("\n") for ln in f]
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f]

def _numpy_loadtxt_path(p) -> np.ndarray:
    """Load text floats via numpy.loadtxt from tk.Path or str."""
    path = p.get_path() if hasattr(p, "get_path") else str(p)
    return np.loadtxt(path)

def _path_str(p) -> str:
    return p.get_path() if hasattr(p, "get_path") else str(p)

def _build_prior_log_for_model_vocab(
    prior_vec: np.ndarray,
    prior_vocab: List[str],
    model_vocab: Optional[List[str]],
    V_expected: int,
) -> np.ndarray:
    """
    If model_vocab is provided, remap prior_vec (ordered by prior_vocab) to model indices.
    Else, assert shapes match and return as-is.
    """
    if model_vocab is None:
        if prior_vec.shape[0] != V_expected:
            raise ValueError(
                f"prior length {prior_vec.shape[0]} != model V {V_expected} and no model_vocab provided for remapping"
            )
        return prior_vec

    if len(model_vocab) != V_expected:
        raise ValueError(f"model_vocab length {len(model_vocab)} != V_expected {V_expected}")

    prior_index = {tok: i for i, tok in enumerate(prior_vocab)}
    remapped = np.full((V_expected,), fill_value=-1e30, dtype=np.float32)  # default very small log-prob
    missing = []
    for j, tok in enumerate(model_vocab):
        i = prior_index.get(tok)
        if i is None:
            # Not found in prior vocab: keep default very small log-prob (acts like masked)
            missing.append(tok)
        else:
            remapped[j] = prior_vec[i]
    # You may log/inspect `missing` if needed.
    return remapped

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
    """RescoreDef API: CTC forced-align + prior rescoring (Prior provided via _other['prior'])."""
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim
    from torchaudio.functional import forced_align

    # --- Forward model to get logits and encoder time dim ---
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    assert isinstance(logits, Tensor) and isinstance(enc_spatial_dim, Dim)
    assert logits.feature_dim is not None and enc_spatial_dim in logits.dims
    log_probs = model.log_probs_wb_from_logits(logits)  # RETURNN Tensor
    emissions: torch.Tensor = log_probs.raw_tensor  # (B,T,V), log-softmaxed

    assert isinstance(emissions, torch.Tensor) and emissions.dim() == 3
    B, T, V = emissions.shape
    device = emissions.device
    dtype = emissions.dtype

    # --- Input lengths from encoder spatial dim (RETURNN dyn size) ---
    input_lengths_rt: Tensor = enc_spatial_dim.dyn_size_ext
    input_lengths: torch.Tensor = input_lengths_rt.raw_tensor  # (B,)
    assert input_lengths.shape == (B,)

    # --- Load Prior from _other['prior'] ---
    prior_obj = _other.get("prior", None)
    if prior_obj is None:
        raise ValueError("Expected `_other['prior']` (Prior dataclass with fields file/type/vocab)")

    # Read numeric vector
    prior_vec = _numpy_loadtxt_path(prior_obj.file).astype(np.float32)  # (len(prior_vocab),)
    # Convert to log space if necessary
    ptype = getattr(prior_obj, "type", "log_prob")
    if ptype == "log_prob":
        prior_log_vec = prior_vec
    elif ptype == "prob":
        # Guard against zeros
        prior_log_vec = np.log(np.maximum(prior_vec, np.finfo(np.float32).tiny))
    else:
        raise ValueError(f"invalid prior.type {ptype!r}; expected 'log_prob' or 'prob'")

    # Read prior vocab
    prior_vocab_path = _path_str(prior_obj.vocab)
    prior_vocab = _read_lines(prior_vocab_path)

    if len(prior_vocab) != prior_log_vec.shape[0]:
        raise ValueError(
            f"prior vocab size {len(prior_vocab)} != prior vector length {prior_log_vec.shape[0]}"
        )

    # Optional: model vocab for remapping (indices must match token IDs in logits/targets)
    model_vocab: Optional[List[str]] = _other.get("model_vocab", None)
    if model_vocab is not None and len(model_vocab) != V:
        raise ValueError(f"model_vocab length {len(model_vocab)} != model V {V}")

    prior_log_vec = _build_prior_log_for_model_vocab(prior_log_vec, prior_vocab, model_vocab, V)

    # Torch tensor on correct device/dtype
    prior_log_t = torch.from_numpy(prior_log_vec).to(device=device, dtype=dtype)

    # --- Prepare dims to return scores with (batch, beam) ---
    rem_dims = targets.remaining_dims(targets_spatial_dim)  # {batch_dim, targets_beam_dim}
    batch_dim = next(d for d in rem_dims if d is not targets_beam_dim)
    assert isinstance(batch_dim, Dim)
    blank_idx = int(model.blank_idx)

    # --- Loop over beam to avoid huge broadcasts ---
    per_beam_scores = []
    for beam_idx in range(targets_beam_dim.get_dim_value()):
        # Slice one beam
        targets_b = rf.gather(targets, axis=targets_beam_dim, indices=beam_idx)
        targets_b_seq_lens = rf.gather(targets_spatial_dim.dyn_size_ext, axis=targets_beam_dim, indices=beam_idx)
        targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{targets_spatial_dim.name}_beam{beam_idx}")
        targets_b, _ = rf.replace_dim(targets_b, in_dim=targets_spatial_dim, out_dim=targets_b_spatial_dim)
        targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)

        tokens: torch.Tensor = targets_b.raw_tensor  # (B, Lc)
        tokens_lengths: torch.Tensor = targets_b_spatial_dim.dyn_size_ext.raw_tensor  # (B,)
        tokens = tokens.to(device=device, dtype=torch.long)
        tokens_lengths = tokens_lengths.to(device=device, dtype=torch.long)

        # Forced alignment: best CTC path per sample (includes blanks)
        with torch.no_grad():
            alignment: torch.Tensor = forced_align(
                emissions=emissions,  # (B, T, V), log-probs
                tokens=tokens,  # (B, Lc)
                input_lengths=input_lengths,  # (B,)
                tokens_lengths=tokens_lengths,  # (B,)
                blank=blank_idx,
            )  # -> (B, T) long

        # Sum log-prior over aligned non-blank frames within valid encoder time
        align_clamped = alignment.clamp_(0, V - 1)
        aligned_prior = prior_log_t[align_clamped]  # (B, T)
        non_blank = (alignment != blank_idx)
        t_idx = torch.arange(T, device=device).unsqueeze(0)
        valid_time = t_idx < input_lengths.unsqueeze(1)

        aligned_prior = torch.where(non_blank & valid_time, aligned_prior, torch.zeros_like(aligned_prior))
        per_sample = aligned_prior.sum(dim=1)  # (B,)
        per_beam_scores.append(per_sample)

    scores_bt = torch.stack(per_beam_scores, dim=1)  # (B, beam)

    # Wrap into RETURNN Tensor with dims (batch, beam)
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
