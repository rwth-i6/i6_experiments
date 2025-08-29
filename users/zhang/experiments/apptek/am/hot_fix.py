# Permute the output logits to match the vocabulary
from ast import literal_eval
from typing import Dict, List, Tuple
import sentencepiece as spm
import torch

MISSING_SP_REAL_TOKEN = "▁mes"   # fixed missing piece
NOISE_TARGET_IN_SP = "<sep>"     # <noise> takes <sep>'s SP slot

def load_py_dict_literal(path: str) -> Dict[str, int]:
    with open(path, "rt", encoding="utf-8") as f:
        return literal_eval(f.read())

def build_reco2sp(spm_path: str, reco_vocab_path: str) -> Tuple[List[int], Dict[str, int]]:
    """
    reco_vocab_path: Python dict literal token->reco_id (your lexicon-ordered vocab).
    Returns:
      reco2sp: length = |reco|, each entry is the SP id to map into, or -1 if none
      sp_tok2id: {token: sp_id}
    """
    reco_tok2id = load_py_dict_literal(reco_vocab_path)
    reco_id2tok = {i: t for t, i in reco_tok2id.items()}

    sp = spm.SentencePieceProcessor(model_file=spm_path)
    Vsp = sp.GetPieceSize()
    sp_tok2id = {sp.IdToPiece(i): i for i in range(Vsp)}

    # required SP ids we’ll use
    if NOISE_TARGET_IN_SP not in sp_tok2id:
        raise RuntimeError(f"SP model has no token {NOISE_TARGET_IN_SP!r}")
    if MISSING_SP_REAL_TOKEN not in sp_tok2id:
        raise RuntimeError(f"SP model has no token {MISSING_SP_REAL_TOKEN!r}")
    sep_sid = sp_tok2id[NOISE_TARGET_IN_SP]
    mas_sid = sp_tok2id[MISSING_SP_REAL_TOKEN]

    # base mapping: exact matches; else -1
    Vreco = len(reco_tok2id)
    reco2sp = [-1] * Vreco
    for rid in range(Vreco):
        tok = reco_id2tok[rid]
        reco2sp[rid] = sp_tok2id.get(tok, -1)

    # overrides
    if "<noise>" in reco_tok2id:
        reco2sp[reco_tok2id["<noise>"]] = sep_sid
    if "<music>" in reco_tok2id:
        reco2sp[reco_tok2id["<music>"]] = mas_sid

    # sanity: avoid accidental duplicate targets (except noise→<sep> by design)
    seen = {}
    for rid, sid in enumerate(reco2sp):
        if sid < 0:
            continue
        if sid in seen and not {"<noise>", NOISE_TARGET_IN_SP} >= {reco_id2tok[rid], reco_id2tok[seen[sid]]}:
            a, b = reco_id2tok[seen[sid]], reco_id2tok[rid]
            raise RuntimeError(f"Two reco ids map to the same SP id {sid}: {a!r} and {b!r}")
        seen[sid] = rid

    return reco2sp, sp_tok2id

def up_project_logits_to_sp(
    logits_reco: torch.Tensor,  # [B,T,Vreco]
    reco2sp: List[int],
    Vsp: int,
) -> torch.Tensor: # [B,T,Vsp]
    """
    Produce logits_sp [B,T,Vsp], filling -1e30 where there is no source column.
    """
    device = logits_reco.device
    dtype = logits_reco.dtype
    reco2sp_t = torch.tensor(reco2sp, device=device, dtype=torch.long)

    logits_sp = torch.full((*logits_reco.shape[:-1], Vsp), -1e30, device=device, dtype=dtype)

    valid = reco2sp_t >= 0
    src_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # reco ids we keep
    tgt_idx = reco2sp_t[valid]                                 # their SP ids

    kept = logits_reco.index_select(-1, src_idx)               # [B,T,K]
    logits_sp.scatter_(-1, tgt_idx.view(1,1,-1).expand_as(kept), kept)
    return logits_sp


RECO2SP, SP_TOK2ID = build_reco2sp("/nas/models/asr/artefacts/subword_units/sentencepiece/ES/2025-04-spm_10240-mbw/10240-nmt_nfkc_cf.spm",
                                   "/nas/models/asr/hzhang/setups/2025-07-20--combined/10240-nmt_nfkc_cf_reordered_reco.returnn")

def permute_logits_16khz_spm10k(logits_reco: torch.Tensor):
    return up_project_logits_to_sp(logits_reco, RECO2SP, len(SP_TOK2ID))