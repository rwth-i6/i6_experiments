__all__ = [
    "get_search_parameters",
    "get_lookahead_options",
]


def get_search_parameters(bp=15.0, bpl=50000, wep=0.5, wepl=10000, lsp=None):
    search_params = {
        "beam-pruning": bp,
        "beam-pruning-limit": bpl,
        "word-end-pruning": wep,
        "word-end-pruning-limit": wepl,
    }
    if lsp is not None:
        search_params["lm-state-pruning"] = lsp
    return search_params


def get_lookahead_options(
    *,
    scale=None,  # 0.4
    hlimit=None,  # bigram == 1
    laziness=None,  # 15
    treecf=None,  # 30
    clow=None,  # 2000
    chigh=None,  # 3000
    minrepr=None,  # 1
):
    lmla_options = {}

    if scale is not None:
        lmla_options["lm_lookahead_scale"] = scale
    if hlimit is not None:
        lmla_options["history_limit"] = hlimit
    if laziness is not None:
        lmla_options["laziness"] = laziness
    if treecf is not None:
        lmla_options["tree_cutoff"] = treecf
    if clow is not None:
        lmla_options["cache_low"] = clow
    if chigh is not None:
        lmla_options["cache_high"] = chigh
    if minrepr is not None:
        lmla_options["minimum_representation"] = minrepr

    return lmla_options
