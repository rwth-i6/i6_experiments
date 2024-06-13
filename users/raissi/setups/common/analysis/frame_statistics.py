#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import itertools as it
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys

import sprint_cache as sc


def load_alignment(allophones_path, alignments_path):
    alignments = []
    f = sc.FileArchive(alignments_path)
    f.setAllophones(allophones_path)
    for i, k in enumerate(f.ft):
        finfo = f.ft[k]
        if 'attrib' not in finfo.name:
            alignment = [(f.allophones[mix], state) for time, mix, state, _ in f.read(finfo.name, 'align')]
            alignments.append(alignment)
    return alignments


def count_hmm_states(alignment):
    label_lengths = list()
    for cur_label, val in it.groupby(alignment):
        label_lengths.append(tuple([cur_label, len(list(val))]))
    return label_lengths


def count_merge_hmm_states(alignment):
    label_lengths = list()
    for cur_label, val in it.groupby(alignment, key=lambda t : t[0]):
        label_lengths.append(tuple([cur_label, len(list(val))]))
    return label_lengths


def count_cart_lengths(count, silence_symbol):
    cart_lengths = []
    hmm0_lengths = []
    hmm1_lengths = []
    hmm2_lengths = []
    for label, length in count:
        if label[0].strip().find(silence_symbol) == 0:
            continue
        cart_lengths.append(length)
        if label[1] == 0:
            hmm0_lengths.append(length)
        if label[1] == 1:
            hmm1_lengths.append(length)
        if label[1] == 2:
            hmm2_lengths.append(length)
    return cart_lengths, hmm0_lengths, hmm1_lengths, hmm2_lengths


def count_phon_lengths(count, silence_symbol):
    phon_lengths = []
    for label, length in count:
        if label[0].strip().find(silence_symbol) == 0:
            continue
        phon_lengths.append(length)
    return phon_lengths


def count_silence(count, silence_symbol):
    if count[0][0][0].strip().find(silence_symbol) == 0:
        sil_begin_length = count.pop(0)[1]
    else:
        sil_begin_length = 0
    if count[-1][0][0].strip().find(silence_symbol) == 0:
        sil_end_length = count.pop(-1)[1]
    else:
        sil_end_length = 0
    sil_middle_length = []
    for label, length in count:
        if label[0].strip().find(silence_symbol) == 0:
            sil_middle_length.append(length)
    return [sil_begin_length], sil_middle_length, [sil_end_length]


def main(allophones_path, alignments_path, output_dir, silence_symbol, non_speech_symbol):
    assert isinstance(silence_symbol, str), (silence_symbol, "is not str type")
    pickle_path = f"{output_dir}/data.pickle"

    alignments       = collections.defaultdict(list)  # dict(int:list)
    alignments_merge = collections.defaultdict(list)
    silence_begin = list()
    silence_end   = list()
    silence_intra = list()
    phoneme_total_lengths = list()
    cart_total_lengths = list()
    hmm0_total_lengths = list()
    hmm1_total_lengths = list()
    hmm2_total_lengths = list()
    sil_hist_begin = collections.defaultdict(int)
    sil_hist_end   = collections.defaultdict(int)
    sil_hist_intra = collections.defaultdict(int)
    cart_hist = collections.defaultdict(int)
    hmm0_hist = collections.defaultdict(int)
    hmm1_hist = collections.defaultdict(int)
    hmm2_hist = collections.defaultdict(int)
    phon_hist = collections.defaultdict(int)
    if not os.path.isfile(pickle_path):
        print("collecting statistics")
        num_alignments = len(alignments_path)
        idx = 0
        for i, ap in enumerate(alignments_path):
            print(f"alignment: {i+1}/{num_alignments}")
            assert os.path.isfile(ap), ap
            alignment = load_alignment(allophones_path, ap)
            for align in alignment:
                alignments[idx] = count = count_hmm_states(align)
                alignments_merge[idx] = count_merge = count_merge_hmm_states(align)

                sil_begin, sil_intra, sil_end = count_silence(count, silence_symbol)

                silence_begin.extend(sil_begin)
                silence_intra.extend(sil_intra)
                silence_end.extend(sil_end)
                sil_hist_begin[sil_begin[0]] += 1
                sil_hist_end[sil_end[0]] += 1
                for i in sil_intra:
                    sil_hist_intra[i] += 1

                cart_lengths, hmm0_lengths, hmm1_lengths, hmm2_lengths = count_cart_lengths(count, non_speech_symbol)
                for cart in cart_lengths:
                    cart_hist[cart] += 1
                for h0 in hmm0_lengths:
                    hmm0_hist[h0] += 1
                for h1 in hmm1_lengths:
                    hmm1_hist[h1] += 1
                for h2 in hmm2_lengths:
                    hmm2_hist[h2] += 1

                phon_lengths = count_phon_lengths(count_merge, non_speech_symbol)
                for phon in phon_lengths:
                    phon_hist[phon] += 1

                phoneme_total_lengths.extend(phon_lengths)
                cart_total_lengths.extend(cart_lengths)
                hmm0_total_lengths.extend(hmm0_lengths)
                hmm1_total_lengths.extend(hmm1_lengths)
                hmm2_total_lengths.extend(hmm2_lengths)

                idx += 1

        results = [alignments, alignments_merge, silence_begin, silence_intra, silence_end, sil_hist_begin, sil_hist_intra, sil_hist_end, cart_total_lengths, hmm0_total_lengths, hmm1_total_lengths, hmm2_total_lengths, phoneme_total_lengths, cart_hist, hmm0_hist, hmm1_hist, hmm2_hist, phon_hist]

        with open(pickle_path, "wb") as out_pickle:
            data_dump = tuple(results)
            pickle.dump(data_dump, out_pickle, protocol=4)  # protocol version4 for python 3.4+ support
    else:
        print("found pickled statistics")
        with open(pickle_path, "rb") as in_pickle:
            results = pickle.load(in_pickle)

    # alignments        : dict[int] = list[cart labels, cart lengths]
    # alignments_merge  : dict[int] = list[phon labels, phon lengths]
    # silence_begin     : list[sil lengths]
    # silence_intra     : list[sil lengths]
    # silence_end       : list[sil lengths]
    # sil_hist_begin    : dict[sil lengths] = occurences
    # sil_hist_intra    : dict[sil lengths] = occurences
    # sil_hist_end      : dict[sil lengths] = occurences
    # cart_lengths      : list[cart labels lengths]
    # hmm0_lengths      : list[hmm state 0 lengths]
    # hmm1_lengths      : list[hmm state 1 lengths]
    # hmm2_lengths      : list[hmm state 2 lengths]
    # phon_lengths      : list[phon labels lengths]
    # cart_hist         : dict[cart labels] = occurences
    # hmm0_hist         : dict[hmm state 0] = occurences
    # hmm1_hist         : dict[hmm state 1] = occurences
    # hmm2_hist         : dict[hmm state 2] = occurences
    # phon_hist         : dict[phon labels] = occurences
    return results


def hist_data_to_dataframe(x_label, y_label, data_dict):
    d_t = collections.defaultdict(list)
    for k, v in sorted(data_dict.items()):
        d_t[x_label].append(k)
        d_t[y_label].append(v)

    df = pd.DataFrame(data=d_t)

    return df


def plot(output_dir, plot_dir, inputs):
    print("calculating averages")
    alignments, alignments_merge, silence_begin, silence_intra, silence_end, sil_hist_begin, sil_hist_intra, sil_hist_end, cart_lengths, hmm0_lengths, hmm1_lengths, hmm2_lengths, phon_lengths, cart_hist, hmm0_hist, hmm1_hist, hmm2_hist, phon_hist = inputs
    # *** stat calculation ***
    num_seqs = len(alignments.keys())
    assert num_seqs == len(silence_begin)
    assert num_seqs == len(silence_end)
    total_num_sil = sum(silence_begin) + sum(silence_intra) + sum(silence_end)
    avg_sil_begin = sum(silence_begin) / num_seqs
    avg_sil_intra = sum(silence_intra) / len(silence_intra) if len(silence_intra) > 0 else 0
    avg_sil_end = sum(silence_end) / num_seqs
    avg_cart = sum(cart_lengths) / len(cart_lengths)
    avg_hmm0 = sum(hmm0_lengths) / len(hmm0_lengths)
    avg_hmm1 = sum(hmm1_lengths) / len(hmm1_lengths)
    avg_hmm2 = sum(hmm2_lengths) / len(hmm2_lengths)
    avg_phon = sum(phon_lengths) / len(phon_lengths)

    total_num_frames = 0
    for _, v in alignments.items():
        for label, length in v:
            total_num_frames += int(length)


    with open(f"{output_dir}/statistics.txt", "wt") as out_stats:
        out_stats.write(f"average silence length at sequence begin: {avg_sil_begin:.2f}\n")
        out_stats.write(f"average silence length intra sequence   : {avg_sil_intra:.2f}\n")
        out_stats.write(f"average silence length at sequence end  : {avg_sil_end:.2f}\n")
        out_stats.write(f"average cart label length               : {avg_cart:.2f}\n")
        out_stats.write(f"average 0. hmm state length             : {avg_hmm0:.2f}\n")
        out_stats.write(f"average 1. hmm state length             : {avg_hmm1:.2f}\n")
        out_stats.write(f"average 2. hmm state length             : {avg_hmm2:.2f}\n")
        out_stats.write(f"average phoneme label length            : {avg_phon:.2f}\n")
        out_stats.write(f"average number of frames per sequence   : {total_num_frames/num_seqs:.2f}\n")
        out_stats.write(f"total number of silence frames          : {total_num_sil:.0f}\n")
        out_stats.write(f"total number of frames                  : {total_num_frames:.0f}\n")
        out_stats.write(f"number of sequences                     : {num_seqs:.0f}\n")


    print("creating plots")
    # *** data to pandas dataframe ***
    sil_begin_dataframe = hist_data_to_dataframe("begin silence lengths", "occurences", sil_hist_begin)
    sil_intra_dataframe = hist_data_to_dataframe("intra silence lengths", "occurences", sil_hist_intra)
    sil_end_dataframe = hist_data_to_dataframe("end silence lengths", "occurences", sil_hist_end)
    cart_dataframe = hist_data_to_dataframe("cart lengths", "occurences", cart_hist)
    hmm0_dataframe = hist_data_to_dataframe("hmm 0 lengths", "occurences", hmm0_hist)
    hmm1_dataframe = hist_data_to_dataframe("hmm 1 lengths", "occurences", hmm1_hist)
    hmm2_dataframe = hist_data_to_dataframe("hmm 2 lengths", "occurences", hmm2_hist)
    phon_dataframe = hist_data_to_dataframe("phon lengths", "occurences", phon_hist)

    # *** plot histogram ***
    sil_begin_dataframe.plot(x="begin silence lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/silence_begin_histogram.png")
    sil_intra_dataframe.plot(x="intra silence lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/silence_intra_histogram.png")
    sil_end_dataframe.plot(x="end silence lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/silence_end_histogram.png")
    cart_dataframe.plot(x="cart lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/cart_label_histogram.png")
    hmm0_dataframe.plot(x="hmm 0 lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/hmm0_histogram.png")
    hmm1_dataframe.plot(x="hmm 1 lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/hmm1_histogram.png")
    hmm2_dataframe.plot(x="hmm 2 lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/hmm2_histogram.png")
    phon_dataframe.plot(x="phon lengths", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/phon_label_histogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="alignment statistics")
    parser.add_argument("allophones", nargs=1, type=str, help="allophone file path")
    parser.add_argument("alignments", nargs="*", type=str, help="alignment file paths")
    parser.add_argument("--root_dir", nargs=1, type=str, help="output directory path")
    parser.add_argument("--sub_dir", nargs=1, type=str, help="plot directory path")
    parser.add_argument("--silence_symbol", nargs=1, type=str, help="which silence symbol to use", default=["[SILENCE]{#+#}@i@f"])
    parser.add_argument("--non_speech_symbol", nargs=1, type=str, help="which non speech symbol to use", default=["["])
    args = parser.parse_args()

    assert len(args.allophones) == 1
    allophones_path = args.allophones[0]
    assert os.path.isfile(allophones_path)

    assert len(args.root_dir) == 1
    root_dir = args.root_dir[0]
    assert len(args.sub_dir) == 1
    sub_dir = args.sub_dir[0]
    output_dir = os.path.join(root_dir, "statistics", sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_dir = os.path.join(root_dir, "plots", sub_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    alignments_path = args.alignments

    assert len(args.silence_symbol) == 1
    silence_symbol = args.silence_symbol[0]

    assert len(args.non_speech_symbol) == 1
    non_speech_symbol = args.non_speech_symbol[0]

    intermediate = main(allophones_path, alignments_path, output_dir, silence_symbol, non_speech_symbol)

    plot(output_dir, plot_dir, intermediate)

