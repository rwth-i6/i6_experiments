__all__ = [
    "get_search_parameters",
    "get_lookahead_options",
]


def get_search_parameters(bp=16.0, bpl=100000, wep=0.5, wepl=15000, lsp=None):
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
    scale=None,
    hlimit=-1,
    laziness=15,
    treecf=30,
    clow=2000,
    chigh=3000,
    minrepr=1,
):
    lmla_options = {
        "history_limit": hlimit,
        "minimum_representation": minrepr,
        "laziness": laziness,
        "tree_cutoff": treecf,
        "cache_low": clow,
        "cache_high": chigh,
    }
    if scale is not None:
        lmla_options["lm_lookahead_scale"] = scale

    return lmla_options
