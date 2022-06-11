import matplotlib.pyplot as plt                                                                                                                                                               
import numpy as np
import json

ex = "baseline_07_big_short"
sub_ex = "baseline_07_big_short+num-blocks-XXX-aux-at-half"

experiments = []
_set = "dev-other"

blocks = list(range(10, 16+1))

for x in range(10, 16+1):
    if x == 12:
        experiments.append(f'{ex}/{ex}')
    else:
        experiments.append(f'{ex}/{sub_ex.replace("XXX", str(x))}')

paths = []

for ex in experiments:
    paths.append(f'results2/{ex}.json')

datas = []

for p in paths:

    with open(p) as data_file:
        data = json.load(data_file)

    datas.append(data)

# epochs on X
# wer on Y

def x_y_for_ex(i):
    X = list(datas[i][_set]["optim_wer_by_ep"].keys())
    Y = [
    float(datas[i][_set]["optim_wer_by_ep"][x][-2][:-1]) for x in X
    ]
    X = [float(x) for x in X]
    return X, Y

plt.xlabel("sub epoch")
plt.ylabel("WER % dev-other")

for i, d in enumerate(datas):
    X, Y = x_y_for_ex(i)
    print(X)
    print(Y)
    plt.plot(X, Y, label=blocks[i])

plt.legend(loc="upper right")

plt.savefig("WERconvergence-num-blocks.pdf")
