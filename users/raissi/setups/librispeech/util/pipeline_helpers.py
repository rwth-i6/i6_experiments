__all__ = ['get_label_info', 'get_alignment_keys', 'get_lexicon_args', 'get_tdp_values']


def get_label_info(
    n_states=3,
    ph_emb_size=32,
    st_emb_size=128,
    use_word_end_class=True,
    use_boundary_classes=False,
    is_folded=True,
):

    return {
        'n_states_per_phone': n_states,
        'n_phonemes': 42 if is_folded else 72,
        'ph_emb_size': ph_emb_size,
        'st_emb_size': st_emb_size,
        'sil_id': None, #toDo: a job that gives the sielnce label within a specific state tying
        'state_tying': 'no-dense-tying',
        'use_word_end_class': use_word_end_class,
        'use_boundary_classes': use_boundary_classes,

    }

def get_alignment_keys(additional_keys=None):
    keys = ['GMMmono', 'GMMtri', 'scratch', 'FHmono', 'FHdi', 'FHtri']
    if additional_keys is not None:
        keys.extend(additional_keys)
    return keys


def get_lexicon_args(add_all_allophones=False, norm_pronunciation=True):
    return {
        'add_all_allophones': add_all_allophones,
        'norm_pronunciation': norm_pronunciation,
    }

def get_tdp_values():
    return {
        'pattern' : ["loop", "forward", "skip", "exit"],
        'default': {'*': (3.0, 0.0, "infinity", 0.0), 'silence': (0.0, 3.0, "infinity", 20.0)}
    }
