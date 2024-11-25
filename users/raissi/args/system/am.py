def get_lexicon_args(add_all_allophones=False, norm_pronunciation=True):
    return {
        "add_all_allophones": add_all_allophones,
        "norm_pronunciation": norm_pronunciation,
    }


def get_tdp_values():
    from math import log

    speech_fwd_three = 0.350  # 3/9 for 3partite
    speech_fwd_mono = 0.125  # 1/8 for phoneme
    silence_fwd = 0.04  # 1/25 following the start/end segment silence

    return {
        "pattern": ["loop", "forward", "skip", "exit"],
        "default": {
            "*": (3.0, 0.0, "infinity", 0.0),
            "silence": (0.0, 3.0, "infinity", 20.0),
        },
        "heuristic": {
            "monostate": {
                "*": (-log(1 - speech_fwd_mono), -log(speech_fwd_mono), "infinity", 0.0),
                "silence": (-log(1 - silence_fwd), -log(silence_fwd), "infinity", 0.0),
            },
            "threepartite": {
                "*": (-log(1 - speech_fwd_three), -log(speech_fwd_three), "infinity", 0.0),
                "silence": (-log(1 - silence_fwd), -log(silence_fwd), "infinity", 0.0),
            },
        },
        "heuristic-30ms": {
            "monostate": {
                "*": (-log(1 - 3 * speech_fwd_mono), -log(3 * speech_fwd_mono), "infinity", 0.0),
                "silence": (-log(1 - 3 * silence_fwd), -log(3 * silence_fwd), "infinity", 0.0),
            },
        },
        "heuristic-40ms": {
            "monostate": {
                "*": (-log(1 - 4 * speech_fwd_mono), -log(4 * speech_fwd_mono), "infinity", 0.0),
                "silence": (-log(1 - 4 * silence_fwd), -log(4 * silence_fwd), "infinity", 0.0),
            },
        },
    }
