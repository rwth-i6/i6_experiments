__all__ = ['get_recog_mono_args', 'get_recog_mono_specAug_args']


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
                 "runOptJob" : True,
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
                 "runOptJob" : True,
                }
    return recog_args

