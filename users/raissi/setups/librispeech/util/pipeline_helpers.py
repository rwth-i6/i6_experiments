__all__ = ['get_label_info', 'get_alignment_keys', 'get_lexicon_args', 'get_tdp_values']


def get_label_info(
    n_states=3,
    ph_emb_size=32,
    st_emb_size=128,
    use_word_end_classes=True,
    use_boundary_classes=False,
    is_folded=True,
):

    return {
        'n_states_per_phone': n_states,
        'n_contexts': 42 if is_folded else 72,
        'ph_emb_size': ph_emb_size,
        'st_emb_size': st_emb_size,
        'sil_id': None, #toDo: a job that gives the sielnce label within a specific state tying
        'state_tying': 'monophone-no-tying-dense', #no-tying-dense for decoding
        'use_word_end_classes': use_word_end_classes,
        'use_boundary_classes': use_boundary_classes,

    }

def get_alignment_keys(additional_keys=None):
    keys = ['GMMmono', 'GMMtri', 'scratch', 'mono', 'FHmono', 'FHdi', 'FHtri']
    if additional_keys is not None:
        keys.extend(additional_keys)
    return keys


def get_lexicon_args(add_all_allophones=False, norm_pronunciation=True):
    return {
        'add_all_allophones': add_all_allophones,
        'norm_pronunciation': norm_pronunciation,
    }

def get_tdp_values():
    from math import log
    speech_fwd_three = 0.350 #3/9 for 3partite
    speech_fwd_mono  = 0.125 #1/8 for phoneme
    silence_fwd      = 0.04 #1/25 following the
    return {
        'pattern' : ["loop", "forward", "skip", "exit"],
        'default': {'*': (3.0, 0.0, "infinity", 0.0), 'silence': (0.0, 3.0, "infinity", 20.0)},
        'heuristic' : {'monostate': {'*': (-log(1-speech_fwd_mono), -log(speech_fwd_mono), "infinity", 0.0),
                                     'silence': (-log(1-silence_fwd), -log(silence_fwd), "infinity", 0.0)},
                       'threepartite':  {'*': (-log(1-speech_fwd_three), -log(speech_fwd_three), "infinity", 0.0),
                                     'silence': (-log(1-silence_fwd), -log(silence_fwd), "infinity", 0.0)}
                       }
    }
