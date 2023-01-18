import i6_core.rasr as rasr

from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextEnum,
    ContextMapper
)
from i6_experiments.users.raissi.setups.common.helpers.network_architectures import (
    make_config
)



def get_extra_config_segment_order(size, extra_config=None):
    if extra_config is None:
        extra_config = rasr.RasrConfig()
    extra_config['*'].segment_order_sort_by_time_length = True
    extra_config['*'].segment_order_sort_by_time_length_chunk_size = size

    return extra_config




# This function is designed by Wei Zhou
def get_learning_rates(lrate=0.001, pretrain=0, warmup=False, warmupRatio=0.1,
                       increase=90, incMinRatio=0.01, incMaxRatio=0.3, constLR=0,
                       decay=90, decMinRatio=0.01, decMaxRatio=0.3, expDecay=False, reset=False):
    # example fine tuning: get_learning_rates(lrate=5e-5, increase=0, constLR=150, decay=60, decMinRatio=0.1, decMaxRatio=1)
    # example for n epochs get_learning_rates(increase=n/2, cdecay=n/2)
    learning_rates = []
    # pretrain (optional warmup)
    if warmup:
        learning_rates += warmup_lrates(initial=lrate * warmupRatio, final=lrate, epochs=pretrain)
    else:
        learning_rates += [lrate] * pretrain
    # linear increase and/or const
    if increase > 0:
        step = lrate * (incMaxRatio - incMinRatio) / increase
        for i in range(1, increase + 1):
            learning_rates += [lrate * incMinRatio + step * i]
        if constLR > 0:
            learning_rates += [lrate * incMaxRatio] * constLR
    elif constLR > 0:
        learning_rates += [lrate] * constLR
    # linear decay
    if decay > 0:
        if expDecay:  # expotential decay (closer to newBob)
            import numpy as np
            factor = np.exp(np.log(decMinRatio / decMaxRatio) / decay)
            for i in range(1, decay + 1):
                learning_rates += [lrate * decMaxRatio * (factor ** i)]
        else:
            step = lrate * (decMaxRatio - decMinRatio) / decay
            for i in range(1, decay + 1):
                learning_rates += [lrate * decMaxRatio - step * i]
    # reset and default newBob(cv) afterwards
    if reset: learning_rates += [lrate]
    return learning_rates


# designed by Wei Zhou
def warmup_lrates(initial=0.0001, final=0.001, epochs=20):
    lrates = []
    step_size = (final - initial) / (epochs - 1)
    for i in range(epochs):
        lrates += [initial + step_size * i]
    return lrates


def get_monophone_returnn_config(crnnArgs, crnnCode, ctxEmbSize=10, stateEmbSize=30, focalLossFactor=2.0,
                              labelSmoothing=0.2, nInputs=40, nClasses=47 * 47 * 47 * 3, mlpL2=0.01,
                              finalContextType=None,
                              sprint=True, sharedDeltaEncoder=False):
    contextType = contextEnum(ctxMapper.get_enum(1))
    finalContextType = finalContextType if finalContextType is not None else contextEnum(ctxMapper.get_enum(4))

    monoCrnnConfig = make_config(contextType,
                                 ctxMapper,
                                 addMLPs=True,
                                 finalContextType=finalContextType,
                                 extraPython=crnnCode,
                                 sprint=sprint,
                                 isBoundary=False,
                                 ctxEmbSize=ctxEmbSize,
                                 stateEmbSize=stateEmbSize,
                                 focalLossFactor=focalLossFactor,
                                 labelSmoothing=labelSmoothing,
                                 mlpL2=mlpL2,
                                 sharedDeltaEncoder=sharedDeltaEncoder,
                                 **crnnArgs)

    monoCrnnConfig.config["network"]["center-output"]["target"] = "centerState"
    monoCrnnConfig.config["network"]["left-output"]["target"] = "pastLabel"
    monoCrnnConfig.config["network"]["right-output"]["target"] = "futureLabel"
    monoCrnnConfig.config["num_outputs"] = {"data": [nInputs, 2],
                                            "classes": [nClasses, 1]}
    return monoCrnnConfig


def get_monophone_for_bw_returnn_config(num_classes=126, addMLPs=False, mlpL2=0.0001,
                                        finalContextType=None, sprint=False, **returnn_args):
    ctxMapper = ContextMapper()
    contextType = ContextEnum(ctxMapper.get_enum(1))
    finalContextType = finalContextType if finalContextType is not None else ContextEnum(ctxMapper.get_enum(4))

    monoCrnnConfig = make_config(contextType,
                                 ctxMapper,
                                 addMLPs=addMLPs,
                                 finalContextType=finalContextType,
                                 sprint=sprint,
                                 mlpL2=mlpL2,
                                 **returnn_args)


    for attribute in ['loss', 'loss_opts', 'target']:
        del monoCrnnConfig.config["network"]["center-output"][attribute]
    monoCrnnConfig.config["num_outputs"] = {"data": [returnn_args['num_input'], 2],
                                            "classes": [num_classes, 1]}

    return monoCrnnConfig