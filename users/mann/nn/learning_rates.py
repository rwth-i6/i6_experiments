__all__ = ["get_learning_rates"]

def get_learning_rates(lrate=0.001, pretrain=0, warmup=False, warmup_ratio=0.1,
                       increase=90, inc_min_ratio=0.01, inc_max_ratio=0.3, const_lr=0,
                       decay=90, dec_min_ratio=0.01, dec_max_ratio=0.3, exp_decay=False, reset=False):
    #example fine tuning: get_learning_rates(lrate=5e-5, increase=0, constLR=150, decay=60, decMinRatio=0.1, decMaxRatio=1)
    #example for n epochs get_learning_rates(increase=n/2, cdecay=n/2)
    learning_rates = []
    # pretrain (optional warmup)
    if warmup:
        learning_rates += warmup_lrates(initial=lrate * warmup_ratio, final=lrate, epochs=pretrain)
    else:
        learning_rates += [lrate] * pretrain
    # linear increase and/or const
    if increase > 0:
        step = lrate * (inc_max_ratio - inc_min_ratio) / increase
        for i in range(1, increase + 1):
            learning_rates += [lrate * inc_min_ratio + step * i]
        if const_lr > 0:
            learning_rates += [lrate * inc_max_ratio] * const_lr
    elif const_lr > 0:
        learning_rates += [lrate] * const_lr
    # linear decay
    if decay > 0:
        if exp_decay:  # expotential decay (closer to newBob)
            import numpy as np
            factor = np.exp(np.log(dec_min_ratio / dec_max_ratio) / decay)
            for i in range(1, decay + 1):
                learning_rates += [lrate * dec_max_ratio * (factor ** i)]
        else:
            step = lrate * (dec_max_ratio - dec_min_ratio) / decay
            for i in range(1, decay + 1):
                learning_rates += [lrate * dec_max_ratio - step * i]
    # reset and default newBob(cv) afterwards
    if reset: learning_rates += [lrate]
    return learning_rates
