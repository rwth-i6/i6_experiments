from .masking import *
from .torch_init import *

def write_log_probs_to_files(
    log_probs: np.array,
    txt_path: str = "../output/prior.txt",
    xml_path: str = "../output/prior.xml",
    plot_path: str = "../output/prior.png",
):
    """
    Given a log probs array, write it to .txt, .xml,
    and a plot. Credit: Simon.
    """
    log_prob_strings = ["%.20e" % s for s in log_probs]

    # Write txt file
    with open(txt_path, "wt") as f:
        f.write(" ".join(log_prob_strings))

    # Write xml file
    with open(xml_path, "wt") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="{len(log_probs)}">\n')
        f.write(" ".join(log_prob_strings))
        f.write("\n</vector-f32>")

    # Plot png file
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = np.exp(log_probs)
    xdata = range(len(probs))
    plt.semilogy(xdata, probs)
    plt.xlabel("emission idx")
    plt.ylabel("prior")
    plt.grid(True)
    plt.savefig(plot_path)
