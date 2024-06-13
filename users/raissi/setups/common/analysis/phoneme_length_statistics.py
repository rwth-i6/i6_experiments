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

from frame_statistics import load_alignment


def get_phoneme(label):
    return label.strip().split("{")[0]


def main(allophones_path, alignments_path, output_dir):
    pickle_path = f"{output_dir}/data2.pickle"
    
    single_phoneme_lengths = collections.defaultdict(list)
    long_sequences = list()

    if not os.path.isfile(pickle_path):
        print("collecting statistics")
        num_alignments = len(alignments_path)
        for i, ap in enumerate(alignments_path):
            print(f"alignment: {i+1}/{num_alignments}")
            assert os.path.isfile(ap), ap
            alignment = load_alignment(allophones_path, ap)
            for align in alignment:
                for cur_label, val in it.groupby(align, key=lambda t : t[0]):
                    single_phon = get_phoneme(cur_label)
                    label_length = len(list(val))
                    single_phoneme_lengths[single_phon].append(label_length)
                    if label_length >= 50:
                        long_sequences.append(tuple([single_phon, label_length, align]))

        results = [single_phoneme_lengths, long_sequences]

        with open(pickle_path, "wb") as out_pickle:
            data_dump = tuple(results)
            pickle.dump(data_dump, out_pickle, protocol=4)  # protocol version4 for python 3.4+ support
    else:
        print("found pickled statistics")
        with open(pickle_path, "rb") as in_pickle:
            results = pickle.load(in_pickle)

    return results


def hist_data_to_dataframe(x_label, y_label, data_dict):
    d_t = collections.defaultdict(list)
    for k, v in sorted(data_dict.items()):
        d_t[x_label].append(k)
        d_t[y_label].append(v)

    df = pd.DataFrame(data=d_t)

    return df


def plot(output_dir, plot_dir, inputs):
    long_sequences_path = f"{output_dir}/long_sequences.txt"
    print("calculating averages")
    single_phoneme_lengths, long_sequences = inputs
    # *** dump stats ***
    with open(long_sequences_path, "wt") as out_seqs:
        for seq in long_sequences:
            out_seqs.write(f"{seq}\n\n")

    with open(f"{output_dir}/phoneme_lengths.txt", "wt") as out_stats:
        out_stats.write("average phoneme length")
        for k, v in sorted(single_phoneme_lengths.items()):
            avg = sum(v)/len(v)
            out_stats.write(f"{k}: {avg:.2f}\n")


    print("creating plots")
    # *** data to pandas dataframe ***
    for label, lengths in sorted(single_phoneme_lengths.items()):
        hist = collections.defaultdict(int)
        for l in lengths:
            hist[l] += 1

        phon_df = hist_data_to_dataframe(f"phoneme label lengths {label}", "occurences", hist)

        # *** plot histogram ***
        phon_df.plot(x=f"phoneme label lengths {label}", y="occurences", logy=True).get_figure().savefig(f"{plot_dir}/phoneme_{label}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="alignment statistics")
    parser.add_argument("allophones", nargs=1, type=str, help="allophone file path")
    parser.add_argument("alignments", nargs="*", type=str, help="alignment file paths")
    parser.add_argument("--root_dir", nargs=1, type=str, help="output directory path")
    parser.add_argument("--sub_dir", nargs=1, type=str, help="plot directory path")
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

    intermediate = main(allophones_path, alignments_path, output_dir)

    plot(output_dir, plot_dir, intermediate)

