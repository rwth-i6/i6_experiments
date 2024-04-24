__all__=["get_init_am_args"]

def get_init_am_args():
    return {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),  # loop, forward, skip, exit
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global",
        "nonword_phones": "",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            21.0,
        ),  # only used when tying_type = global-and-nonword
    }