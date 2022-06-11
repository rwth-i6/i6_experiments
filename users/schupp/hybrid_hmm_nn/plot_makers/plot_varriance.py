# This script creates some plots on training varriance:
import matplotlib.pyplot as plt                                                                                                                                                               
import numpy as np
import json

RES = "results2"

dataset = "dev-clean"

with_random_seed = False

# THis is with fixed seed
if not with_random_seed:
    experiments = [
    f"baseline_05_big_short/baseline_05_big_short+random-seed=27-run-{x}" for x in range(1, 5 + 1)
    ]
else:
    # This is with random seed
    experiments = [
        "baseline_05_big_short/baseline_05_big_short+fixed-seed=1490-run-1",
        "baseline_05_big_short/baseline_05_big_short+fixed-seed=3218-run-2", "baseline_05_big_short/baseline_05_big_short+fixed-seed=5537-run-3",
        "baseline_05_big_short/baseline_05_big_short+fixed-seed=1814-run-4",
        "baseline_05_big_short/baseline_05_big_short+fixed-seed=13-run-5"
    ]

# What should be in this plot?
# varriance of WER on dev-other
# Y axis: WER rate
# X axis: epoch X


data_by_ex = {}
for ex in experiments:
    with open(f"{RES}/{ex}.json") as data_file:
        data = json.load(data_file)

    data_by_ex[ex] = data

all_epochs = sorted([ int(x) for x in list(data_by_ex[experiments[0]][dataset]["optim_wer_by_ep"].keys()) ])
print(all_epochs)

# Now add one separte plot for *all* experiments
ex_plots = {}
for ex in experiments:
    ex_plots[ex] = {
        "X" : all_epochs.copy(),
        "Y" : [ float(data_by_ex[ex][dataset]["optim_wer_by_ep"][str(ep)][-2].replace("%", "")) for ep in all_epochs ]
    }

plt.xlabel("sub epoch")
plt.ylabel("WER % dev-other")

for ex in experiments:
    plt.plot(ex_plots[ex]["X"], ex_plots[ex]["Y"])

ep_var = []
ep_spread = []
wer_average = []
ep_min_max = []
for ep in all_epochs:
    all_wers_by_ep = [
            float(data_by_ex[ex][dataset]["optim_wer_by_ep"][str(ep)][-2].replace("%", "")) for ex in experiments ]

    wer_average.append(
        np.average(all_wers_by_ep)
    )

    ep_var.append(np.var(
        all_wers_by_ep
    ))

    ep_spread.append(
        np.max(all_wers_by_ep) - np.min(all_wers_by_ep)
    )

    ep_min_max.append(
        [np.min(all_wers_by_ep) , np.max(all_wers_by_ep)]
    )

print(ep_var)
print(ep_spread)
print(wer_average)
print(ep_min_max)

# Then we also want to plot error bars for everything
#plt.errorbar(all_epochs, )

if dataset == "dev-other":
    plt.errorbar(all_epochs, wer_average, yerr=ep_spread, capsize=4)

    if with_random_seed:
        plt.savefig("grafic-wer-varriance-error-different-seed.pdf")
    else:
        plt.savefig("grafic-wer-varriance-error-same-seed.pdf")

    def get_ex_plot_vals(): # TO interface with jupyter note book
        return ex_plots, experiments
    


