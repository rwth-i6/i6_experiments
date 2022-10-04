__all__ = ['get_recog_mono_args', 'get_recog_mono_specAug_args', 'get_recog_diphone_fromGmm_specAug_args']


def get_recog_mono_args(beam=20, beamLimit=500000, priorScale=0.6, lmScale=4.0,
                           tdpScale=0.5, silExitExit=20.0, tdpExit=0.0):
    recog_args = {"beam": beam,
                 "beamLimit": beamLimit,
                 "priorScales": {'center-state' : priorScale},
                 "pronScale": 3.0,
                 "tdpExit": tdpExit,
                 "silExit": silExitExit,
                 "tdpScale": tdpScale,
                 "lmScale": lmScale,
                }
    return recog_args

def get_recog_mono_specAug_args(beam=20, beamLimit=500000, priorScale=0.3, lmScale=3.0,
                                 tdpScale=0.1, silExitExit=20.0, tdpExit=0.0):
    recog_args = {"beam": beam,
                 "beamLimit": beamLimit,
                 "priorScales": {'center-state' : priorScale},
                 "pronScale": 3.0,
                 "tdpExit": tdpExit,
                 "silExit": silExitExit,
                 "tdpScale": tdpScale,
                 "lmScale": lmScale,
                }
    return recog_args


def get_recog_diphone_fromGmm_specAug_args(beam=20, beamLimit=1000000, centerPrior=0.5, ctxPrior=0.4,
                                           lmScale=5.0, tdpScale=0.5, silExitExit=20.0, tdpExit=0.0):
    return {
        "shared_args": {"beam": beam,
                            "beamLimit": beamLimit,
                            "priorScales": {'center-state':centerPrior, 'left-context': ctxPrior},
                            "pronScale": 3.0,
                            "tdpExit": tdpExit,
                            "silExit": silExitExit,
                            "tdpScale": tdpScale,

                            },

        "4gram_args": {"lmScale": lmScale},

        "lstm_args": {"lstmLmScale": 8.0,  # check grid search
                          "separate_lookahead_scale": 6.0,
                          # "lookahead_history_limit"      : 1,
                          "sparse_lookahead": False,
                          # "recombination_limit": 12,
                          },

    }
