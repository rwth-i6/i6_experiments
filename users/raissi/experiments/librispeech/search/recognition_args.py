__all__ = ['get_recog_mono_args',
           'get_recog_diphone_args'           
           'get_recog_triphone_args']


def get_recog_mono_args(beam=20, beamLimit=500000, priorScale=0.3, lmScale=3.0,
                                 tdpScale=0.1, silExitExit=20.0, tdpExit=0.0):
    recog_args = {"beam": beam,
                 "beamLimit": beamLimit,
                 "priorScales": {'center-state' : priorScale},
                 "pronScale": 3.0,
                 "tdpExit": tdpExit,
                 "silExit": silExitExit,
                 "tdpScale": tdpScale,
                 "lmScale": lmScale,
                 "runOptJob" : True,
                }
    return recog_args


def get_recog_diphone_args(beam=20, beamLimit=500000, centerPrior=0.3, leftCtxPrior=0.1,
                                    lmScale=7.0, tdpScale=0.3, silExit=15.0, tdpExit=0.0):
    return {
        "shared_args": {"beam": beam,
                            "beamLimit": beamLimit,
                            "priorScales": {'left-context': leftCtxPrior, 'center-state': centerPrior},
                            "pronScale": 3.0,
                            "tdpExit": tdpExit,
                            "silExit": silExit,
                            "tdpScale": tdpScale,

                            },

        "4gram_args": {"lmScale": lmScale},

        "lstm_args": {"lstmLmScale"             : lmScale+2.0,
                      "separate_lookahead_scale": 6.0,
                      "lookahead_history_limit" : 1,
                      "sparse_lookahead"        : False,
                      # "recombination_limit": 12,
                    },
    }


def get_recog_triphone_args(beam=20, beamLimit=400000, triPrior=0.1, dpPrior=0.3, ctxPrior=0.1,
                               lmScale=10.0, tdpScale=0.5, silExitExit=15.0, tdpExit=0.0, pronScale=4.0, lstm=9.0,
                               la=10.0):
    return {
        "sharedRecogArgs": {"beam": beam,
                            "beamLimit": beamLimit,
                            "priorScales": {'right-context': triPrior, 'center-state': dpPrior,
                                            "left-context": ctxPrior},
                            "pronScale": pronScale,
                            "tdpExit": tdpExit,
                            "silExit": silExitExit,
                            "tdpScale": tdpScale,
                            },

        "recogArgsCount": {"lmScale": lmScale},

        "recogArgsLstm": {"lstmLmScale": lstm,  # check grid search
                          "separate_lookahead_scale": la,
                          # "lookahead_history_limit": 1,
                          "sparse_lookahead": False,
                          # "recombination_limit"      : 12,
                          }
    }
