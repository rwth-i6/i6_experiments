import numpy as np

from i6_experiments.users.mueller.experiments.ctc_baseline.model import Model

def convert_to_output_hyps(model: Model, hyp: list) -> list:
    prev = None
    ls = []
    for h in hyp:
        if h != prev:
            ls.append(h)
            prev = h
    ls = [h for h in ls if h != model.blank_idx]
    return ls

def hyps_ids_to_label(model: Model, hyp: list, remove_reps_and_blanks: bool = False) -> list:
    if remove_reps_and_blanks:
        hyp = convert_to_output_hyps(model, hyp)
    out_hyp = [model.target_dim.vocab.id_to_label(h) for h in hyp]
    out_hyp = " ".join(out_hyp).replace("@@ ", "")
    if out_hyp.endswith("@@"):
        out_hyp = out_hyp[:-2]
    return out_hyp

def plot_grad(gradients, title):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from datetime import datetime
    
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.supylabel("Vocab")
    fig.supxlabel("Timestep")
    
    
    # ax.imshow(gradients.T, origin="lower", cmap=cm.gray)
    # ax.set_yticks(np.arange(0, 185, 10))
    # ax.set_title("Gradients " + title)
    # g_min = -1 * gradients.min()
    # g_max = -1 * gradients.max()
    # ax.text(2, -20, f'black: 1.0, white: 0.0', bbox={'facecolor': 'white', 'pad': 10})
    
    # now = datetime.now()
    # fig.savefig("/u/marten.mueller/dev/ctc_baseline/output/plots/gradients" + now.strftime("_%H:%M:%S_%d-%m") + ".png")
    
    log_gr = np.log((-gradients))
    ax.imshow(-log_gr.T, origin="lower", cmap=cm.gray)
    ax.set_yticks(np.arange(0, 185, 10))
    ax.set_title("Log Gradients " + title)
    ax.text(2, -20, f'black: {log_gr.max()}, white: {log_gr.min()}', bbox={'facecolor': 'white', 'pad': 10})
    
    now = datetime.now()
    fig.savefig("/u/marten.mueller/dev/ctc_baseline/output/plots/log_gradients" + now.strftime("_%H:%M:%S_%d-%m") + ".png")