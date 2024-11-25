import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.scales import *

import os

# figure.figsize : 4.9, 3.5
# font.size:    11.0
# font.family: serif
# font.serif: Palatino
# axes.titlesize: medium
# figure.titlesize: medium
# text.usetex: True
# text.latex.preamble: \usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]

mpl.rcParams['figure.figsize'] = 8.0, 2.5
mpl.rcParams['font.size'] = 11.0
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]'

model_names = scales_model_names_exclude_edge
train_scales = [train_scale[model_name] for model_name in model_names]
xticks = [0.1, 0.3, 0.5, 0.7, 0.9]

test_scores = {
    "mc": {
        "no_lm": {
            "opls": np.asarray([scales_att_ctc_opls[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_optsr[model_name]["wer"][1] for model_name in model_names]),
            "att": np.asarray([scales_att[model_name]["wer"][1] for model_name in model_names]),
            "ctc": np.asarray([scales_ctc_prior_opls[model_name]["wer"][1] for model_name in model_names]),
        },
        "with_lm": {
            "opls": np.asarray([scales_att_ctc_lm_opls[model_name]["wer"][1] for model_name in model_names]),
            "att": np.asarray([scales_att_lm[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_lm_optsr[model_name]["wer"][1] for model_name in model_names]),
            "ctc": np.asarray([scales_ctc_lm_opls[model_name]["wer"][1] for model_name in model_names]),
        }
    },
    "sc": {
        "no_lm": {
            "opls": np.asarray([scales_att_ctc_only_opls[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_only_optsr[model_name]["wer"][1] for model_name in model_names]),
        },
        "with_lm": {
            "opls": np.asarray([scales_att_ctc_only_lm_opls[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_only_lm_optsr[model_name]["wer"][1] for model_name in model_names]),
        },
    }
}

def make_wer_plot_att_ctc_scales(plot_numbers=None):
    #remove model prefix from model names

    plot_name = "att_ctc_scales"

    fig, axs = plt.subplots(1, 2, sharey=True)
    # plt.xticks(rotation=45, ha="right")

    # model combination no lm
    axs[0].set_title('No LM')
    line1, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["ctc"], label='CTC', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line2, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["att"], label='Attention', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3

    axs[0].set_yticks(np.arange(6.0, 9.5, 0.5))
    axs[0].set_xticks(xticks)

    axs[0].grid(axis='y')
    axs[0].set_ylabel('WER[\%] ($\leftarrow$)')  # Add a y-label to the axes.
    axs[0].set_xlabel('training scale $\lambda$')

    # model combination with lm
    axs[1].set_title('With LM')
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["ctc"], label='CTC', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["att"], label='Attention', marker='v', linewidth=1.0)
    axs[1].set_yticks(np.arange(6.0, 9.5, 0.5))
    axs[1].set_xticks(xticks)
    axs[1].grid(axis='y')
    # axs[1].set_ylabel('WER[\%] ($\leftarrow$)')
    axs[1].set_xlabel('training scale $\lambda$')
    axs[1].legend(handles=[line2, line1])
    axs[1].tick_params(axis='y', which='both', left=True, labelleft=True)

    # fig.supxlabel('train scale $\lambda$')

    # plt.show()

    # save plot
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

make_wer_plot_att_ctc_scales()

def make_wer_plot_att_ctc_scales_joint_decoding_mc():
    #remove model prefix from model names

    plot_name = "att_ctc_scales_joint_decoding_mc"

    fig, axs = plt.subplots(1, 2, sharey=True)

    # model combination no lm
    axs[0].set_title('No LM')
    line1, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["opls"], label='Label-sync.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line2, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["optsr"], label='Time-sync.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line3, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["att"], label='Attention', marker='x', linewidth=0.8) # linestyle=':', linewidth=0.5, marker='o', markersize=3

    axs[0].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[0].set_xticks(xticks)
    axs[0].grid(axis='y')
    axs[0].set_ylabel('WER[\%] ($\leftarrow$)')  # Add a y-label to the axes.
    axs[0].set_xlabel('training scale $\lambda$')

    # model combination with lm
    axs[1].set_title('With LM')
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["opls"], label='ls', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["optsr"], label='ts max', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["att"], label='ATT', marker='x', linewidth=0.8)
    axs[1].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[1].set_xticks(xticks)
    axs[1].grid(axis='y')
    axs[1].legend(handles=[line3, line2, line1])
    axs[1].set_xlabel('training scale $\lambda$')
    axs[1].tick_params(axis='y', which='both', left=True, labelleft=True)


    # plt.show()

    # save plot
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

make_wer_plot_att_ctc_scales_joint_decoding_mc()

def make_wer_plot_att_ctc_scales_joint_decoding_sc():
    #remove model prefix from model names

    plot_name = "att_ctc_scales_joint_decoding_sc"

    fig, axs = plt.subplots(1, 2, sharey=True)

    # model combination no lm
    axs[0].set_title('No LM')
    line1, = axs[0].plot(train_scales, test_scores["sc"]["no_lm"]["opls"], label='Label-sync.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line2, = axs[0].plot(train_scales, test_scores["sc"]["no_lm"]["optsr"], label='Time-sync.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line3, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["att"], label='Attention', marker='x', linewidth=0.8) # linestyle=':', linewidth=0.5, marker='o', markersize=3

    axs[0].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[0].set_xticks(xticks)
    axs[0].grid(axis='y')
    axs[0].set_ylabel('WER[\%] ($\leftarrow$)')  # Add a y-label to the axes.
    axs[0].set_xlabel('training scale $\lambda$')

    # model combination with lm
    axs[1].set_title('With LM')
    axs[1].plot(train_scales, test_scores["sc"]["with_lm"]["opls"], label='ls', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["sc"]["with_lm"]["optsr"], label='ts max', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["att"], label='ATT', marker='x', linewidth=0.8)
    axs[1].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[1].set_xticks(xticks)
    axs[1].grid(axis='y')
    axs[1].legend(handles=[line3, line2, line1])
    axs[1].set_xlabel('training scale $\lambda$')
    axs[1].tick_params(axis='y', which='both', left=True, labelleft=True)


    # plt.show()

    # save plot
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

make_wer_plot_att_ctc_scales_joint_decoding_sc()

def make_wer_plot_att_ctc_scales_joint_decoding_sc_ls():
    #remove model prefix from model names

    plot_name = "att_ctc_scales_sc_ls"

    fig, axs = plt.subplots(1, 2, sharey=True)

    # model combination no lm
    axs[0].set_title('No LM')
    line1, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["opls"], label='Model comb.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line2, = axs[0].plot(train_scales, test_scores["sc"]["no_lm"]["opls"], label='System comb.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line3, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["att"], label='Attention', marker='x', linewidth=0.8) # linestyle=':', linewidth=0.5, marker='o', markersize=3

    axs[0].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[0].set_xticks(xticks)
    axs[0].grid(axis='y')
    axs[0].set_ylabel('WER[\%] ($\leftarrow$)')  # Add a y-label to the axes.
    axs[0].set_xlabel('training scale $\lambda$')

    # model combination with lm
    axs[1].set_title('With LM')
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["opls"], label='MC', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["sc"]["with_lm"]["opls"], label='SC', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["att"], label='ATT', marker='x', linewidth=0.8)
    axs[1].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[1].set_xticks(xticks)
    axs[1].grid(axis='y')
    axs[1].legend(handles=[line3, line1, line2])
    axs[1].set_xlabel('training scale $\lambda$')
    axs[1].tick_params(axis='y', which='both', left=True, labelleft=True)

    # save plot
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

make_wer_plot_att_ctc_scales_joint_decoding_sc_ls()

def make_wer_plot_att_ctc_scales_joint_decoding_sc_ts():
    #remove model prefix from model names

    plot_name = "att_ctc_scales_sc_ts"

    fig, axs = plt.subplots(1, 2, sharey=True)

    # model combination no lm
    axs[0].set_title('No LM')
    line1, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["optsr"], label='Model comb.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line2, = axs[0].plot(train_scales, test_scores["sc"]["no_lm"]["optsr"], label='System comb.', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3
    line3, = axs[0].plot(train_scales, test_scores["mc"]["no_lm"]["att"], label='Attention', marker='x', linewidth=0.8) # linestyle=':', linewidth=0.5, marker='o', markersize=3

    axs[0].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[0].set_xticks(xticks)
    axs[0].grid(axis='y')
    axs[0].set_ylabel('WER[\%] ($\leftarrow$)')  # Add a y-label to the axes.
    axs[0].set_xlabel('training scale $\lambda$')

    # model combination with lm
    axs[1].set_title('With LM')
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["optsr"], label='MC', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["sc"]["with_lm"]["optsr"], label='SC', marker='v', linewidth=1.0)
    axs[1].plot(train_scales, test_scores["mc"]["with_lm"]["att"], label='Att.', marker='x', linewidth=0.8)
    axs[1].set_yticks(np.arange(6.0, 8.0, 0.5))
    axs[1].set_xticks(xticks)
    axs[1].grid(axis='y')
    axs[1].legend(handles=[line3, line1, line2])
    axs[1].set_xlabel('training scale $\lambda$')
    axs[1].tick_params(axis='y', which='both', left=True, labelleft=True)

    # save plot
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, f"{plot_name}.pdf"), bbox_inches='tight')

make_wer_plot_att_ctc_scales_joint_decoding_sc_ts()

def make_wer_plot(plot_name, model_names, scales, plot_numbers=None):
    #remove model prefix from model names


    dev_scores = [scales[model_name]["wer"][0] for model_name in model_names]
    test_scores = [scales[model_name]["wer"][1] for model_name in model_names]
    train_scales = [train_scale[model_name] for model_name in model_names]

    dev_scores = np.asarray(dev_scores)
    test_scores = np.asarray(test_scores)

    fig, ax = plt.subplots()
    ax.plot(train_scales, dev_scores, label='dev', marker='s', linewidth=1.0)
    ax.plot(train_scales, test_scores, label='test', marker='v', linewidth=1.0) # linestyle=':', linewidth=0.5, marker='o', markersize=3

    if plot_numbers:
        for number, name in plot_numbers:
            print(number)
            ax.hlines(y=number, xmin=0, xmax=9 ,color='r', linestyle='-', label=f'{name}')

    # plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    # ax.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax.set_ylabel('WER')  # Add a y-label to the axes.
    ax.set_xlabel('lambda')  # Add a y-label to the axes.
    ax.set_title(f"WER Tedlium2 {plot_name}")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()

    # save plot
    # plt.savefig(f"{plot_name}.png", bbox_inches='tight')

# make_wer_plot("att_no_lm", scales_att_model_names_exclude_edge, scales_att)
# make_wer_plot("ctc_no_lm", scales_ctc_model_names_exclude_edge, scales_ctc_prior)
# make_wer_plot("ctc_opls_no_lm", scales_ctc_model_names_exclude_edge, scales_ctc_prior_opls)
# make_wer_plot("att_ctc_opls_no_lm", scales_model_names_exclude_edge, scales_att_ctc_opls)
# make_wer_plot("att_ctc_optsr_no_lm", scales_model_names_exclude_edge, scales_att_ctc_optsr)