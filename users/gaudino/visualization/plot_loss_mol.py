import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib.ticker import ScalarFormatter, MaxNLocator

import os

mpl.rcParams['figure.figsize'] = 8, 5
mpl.rcParams['font.size'] = 11.0
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]'

def EpochData(learningRate, error):
  d = {}
  d['learning_rate'] = learningRate
  d.update(error)
  return d

def extract_loss_values(lr_file, loss_key='dev_loss_ce'):
    # out_fname = os.path.join('scores', train_dir.split('/')[-1] + '.train.info.txt')

    with open(lr_file) as f:
        data = eval(f.read())

    # keys = sorted(set(sum([list(info.keys()) for info in data.values()], [])))
    # with open(out_fname, 'w') as out_f:
    #     for key in keys:
    #         for epoch, info in sorted(data.items()):
    #             if key not in info:
    #                 continue
    #             out_f.write("epoch %3i %s: %s\n" % (epoch, key, info[key]))

    breakpoint()

    res = []

    for key in range(1,1000):
        res.append(data[key][loss_key])


    return np.array(res)

def plot_loss_mol():

    plot_name = "plot_loss_mol"
    dir_path = os.path.dirname(os.path.realpath(__file__))

    mol_values = extract_loss_values(os.path.join(dir_path, "lr_file_mol.txt"))
    att_only_values = extract_loss_values(os.path.join(dir_path, "lr_file_att_only.txt"))

    epochs = np.array(range(len(mol_values))) / 20

    fig, ax = plt.subplots()
    ax.plot(epochs, mol_values, label='Multi-task learning') # linestyle=':', linewidth=0.5, marker='o', markersize=3
    ax.plot(epochs, att_only_values, label='Attention only') # linestyle=':', linewidth=0.5, marker='o', markersize=3
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks([1, 10])
    ax.set_xlim(left=0, right=len(mol_values)/20)

    fsize = 12

    # plt.xticks(rotation=45, ha="right")
    # plt.grid(axis='y')
    # ax.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax.set_ylabel('cross entropy dev score', fontsize=fsize)  # Add a y-label to the axes.
    ax.set_xlabel('epoch', fontsize=fsize)  # Add a y-label to the axes.
    # ax.set_title(f"Tedlium2 {plot_name}")  # Add a title to the axes.
    ax.legend(fontsize=fsize)  # Add a legend.


    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

def plot_loss_att_mol_ted2():

    plot_name = "plot_loss_att_mol_ted2"
    dir_path = os.path.dirname(os.path.realpath(__file__))

    mol_values = extract_loss_values(os.path.join(dir_path, "lr_file_mol_ted2.txt"), "dev_score_output/output_prob")
    att_only_values = extract_loss_values(os.path.join(dir_path, "lr_file_att_only_ted2.txt"), "dev_score")

    epochs = np.array(range(len(mol_values))) / 4

    fig, ax = plt.subplots()
    ax.plot(epochs, mol_values, label='Multi-task learning') # linestyle=':', linewidth=0.5, marker='o', markersize=3
    ax.plot(epochs, att_only_values, label='Attention only') # linestyle=':', linewidth=0.5, marker='o', markersize=3
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks([1, 10])
    ax.set_xlim(left=0, right=len(mol_values)/4)

    fsize = 12

    # plt.xticks(rotation=45, ha="right")
    # plt.grid(axis='y')
    # ax.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax.set_ylabel('cross entropy dev score', fontsize=fsize)  # Add a y-label to the axes.
    ax.set_xlabel('epoch', fontsize=fsize)  # Add a y-label to the axes.
    # ax.set_title(f"Tedlium2 {plot_name}")  # Add a title to the axes.
    ax.legend(fontsize=fsize)  # Add a legend.


    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

def plot_loss_ctc_mol_ted2():

    plot_name = "plot_loss_ctc_mol_ted2"
    dir_path = os.path.dirname(os.path.realpath(__file__))

    mol_values = extract_loss_values(os.path.join(dir_path, "lr_file_mol.txt"), "dev_score_ctc")
    att_only_values = extract_loss_values(os.path.join(dir_path, "lr_file_ctc_only.txt"), "dev_score_ctc")

    epochs = np.array(range(len(mol_values))) / 4

    fig, ax = plt.subplots()
    ax.plot(epochs, mol_values, label='Multi-task learning') # linestyle=':', linewidth=0.5, marker='o', markersize=3
    ax.plot(epochs, att_only_values, label='Attention only') # linestyle=':', linewidth=0.5, marker='o', markersize=3
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticks([1, 10])
    ax.set_xlim(left=0, right=len(mol_values)/4)

    fsize = 12

    # plt.xticks(rotation=45, ha="right")
    # plt.grid(axis='y')
    # ax.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax.set_ylabel('cross entropy dev score', fontsize=fsize)  # Add a y-label to the axes.
    ax.set_xlabel('epoch', fontsize=fsize)  # Add a y-label to the axes.
    # ax.set_title(f"Tedlium2 {plot_name}")  # Add a title to the axes.
    ax.legend(fontsize=fsize)  # Add a legend.


    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

plot_loss_att_mol_ted2()