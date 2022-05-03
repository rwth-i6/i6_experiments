__all__ = ["get_label_info", "get_alignment_keys"]


def get_label_info(
    n_states=3,
    ph_emb_size=32,
    st_emb_size=128,
    word_end_class=True,
    boundary_class=False,
    is_folded=True,
):

    return {
        "n_states": n_states,
        "n_contexts": 42 if is_folded else 72,
        "ph_emb_size": ph_emb_size,
        "st_emb_size": st_emb_size,
        "sil_id": None,
        "word_end_class": word_end_class,
        "boundary_class": boundary_class,
    }

def get_alignment_keys(additional_keys=None):
    keys = ['GMMmono', 'GMMtri', 'scratch', 'FHmono', 'FHdi', 'FHtri']
    if additional_keys is not None:
        keys.extend(additional_keys)
    return keys
